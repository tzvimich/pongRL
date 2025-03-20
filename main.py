import base64
import IPython
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())
  return IPython.display.HTML(tag)


# from IPython.core.magic import register_cell_magic
# @register_cell_magic
# def write_and_run(line, cell):
#     argz = line.split()
#     file = argz[-1]
#     mode = 'w'
#     if len(argz) == 2 and argz[0] == '-a':
#         mode = 'a'
#     with open(file, mode) as f:
#         f.write(cell)
#     get_ipython().run_cell(cell)

import time
import gymnasium as gym
import ale_py
import os
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import random
import numpy as np
import cv2
import base64
import IPython
from config import *

print(ENV_NAME)

render_mode='rgb_array'
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode=render_mode)

def generate_random_game():
    # Create environment
    print("The environment has the following {} actions: {}".\
        format(env.action_space.n, env.unwrapped.get_action_meanings()))
    with imageio.get_writer("video/random_agent.mp4", fps=60) as video:
        terminal = True
        for frame in range(10000):
            if terminal:
                env.reset()
                terminal = False
            # Breakout require a "fire" action (action #1) to start the
            # game each time a life is lost.
            # Otherwise, the agent would sit around doing nothing.
            action = random.choice(range(env.action_space.n))
            # Step action
            _, reward, terminal, truncated, info = env.step(action)
            video.append_data(process_image(env.render()))


# This function can resize to any shape but was built to resize to 84x84
def process_image(image, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    image = image.astype(np.uint8)  # cv2 requires np.uint8
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[34:34+160, :160]  # crop image
    image = cv2.resize(image, shape, interpolation=cv2.INTER_NEAREST)
    image = image.reshape((*shape, 1))
    return image

class PongWrapper(object):
    """
    Wrapper for the environment provided by Openai Gym
    """
    def __init__(self, env, no_op_steps: int = 10, history_length: int = 4):
        self.env = env
        self.no_op_steps = no_op_steps
        self.history_length = 4 # number of frames to put together (we need dynamic to see where the ball is going)
        self.state = None
        
    def reset(self, evaluation: bool = False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when we are in evaluation mode, in this case the agent takes a random number of no-op steps if True.
        """
        self.frame = self.env.reset()
        
        # If in evaluation model, take a random number of no-op steps
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)
        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_image(self.frame[0]), self.history_length, axis=2)

    def step(self, action: int, render_mode=render_mode):
        """
        Arguments:
            action: An integer describe action to take
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns also an np.array with rgb values
        Returns:
            processed_image: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
        """
        new_frame, reward, terminal, truncated, info = self.env.step(action)
        processed_image = process_image(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_image, axis=2) # replace the first observation of the previous state with the last one
        if render_mode == 'rgb_array':
            return processed_image, reward, terminal, self.env.render()
        elif render_mode == 'human':
            self.env.render()
        return processed_image, reward, terminal, None


class ReplayBuffer(object):
    """
    Replay Memory that stores the last size transitions
    """
    def __init__(self, size: int=1000000, input_shape: tuple=(84, 84), history_length: int=4, reward_type: str = "integer"):
        """
        Arguments:
            size: Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Number of frames stacked together that the agent can see
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to
        self.reward_type = reward_type
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)
    
    def add_experience(self, action, frame, reward, terminal, clip_reward=True, reward_type="integer"):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')
        if clip_reward:
            if reward_type == "integer":
                reward = np.sign(reward)
            else:
                reward = np.clip(reward, -1.0, 1.0)
        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size # when a < b then a % b = a
    
    def get_minibatch(self, batch_size: int = 32):
        """
        Returns a minibatch of size batch_size
        Arguments:
            batch_size: How many samples to return
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
        """
        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame
                index = random.randint(self.history_length, self.count - 1)
                # We check that all frames are from same episode with the two following if statements.
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)
        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))
        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]
    
    def save(self, folder_name):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)
    def load(self, folder_name):
        """
        Load the replay buffer
        """
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')


import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop # we'll use Adam instead of RMSprop

def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4, hidden=1024):
    """
    Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible actions
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed image
        history_length: Number of historical frames to stack togheter
        hidden: Integer, Number of filters in the final convolutional layer. 
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255, output_shape=(input_shape[0], input_shape[1], history_length))(model_input)  # normalize by 255
    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x) # 84,84 -> 20,20
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x) # 20,20 -> 9,9
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x) # 9,9 -> 7,7
    x = Conv2D(hidden, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x) # 7,7 -> 1,1
    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer
    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)
    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True), output_shape=(1,))  # custom layer to reduce mean

    # adv_mean = tf.reduce_mean(adv, axis=1, keepdims=True)  # (None, 1)
    # adv_subtracted = Subtract()([adv, adv_mean])  # (None, 6)

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    return model


class DDDQN_AGENT(object):
    
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        # Slopes and intercepts for exploration decrease
        # Calculating epsilon based on frame number
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames
        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn
    
    def get_action(self, frame_number, state, evaluation=False):
        """
        Get the appropriate epsilon value from a given frame number and return the action to take (ExplorationExploitationScheduler)
        """
        # Calculate epsilon based on the frame number
        
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            eps = self.slope_2*frame_number + self.intercept_2
        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        return q_vals.argmax()
    
    def update_target_network(self):
        """
        Update the target Q network
        """
        self.target_dqn.set_weights(self.DQN.get_weights())
    
    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """
        Just to simplify the add experience recall function
        """
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)
    
    def learn(self, batch_size, gamma):
        """
        Sample a batch and use it to improve the DQN
        """
        states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(batch_size=self.batch_size)
        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)
        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]
        # Calculate targets (bellman equation)
        target_q = rewards + (gamma*double_q * (1-terminal_flags))
        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions)
            one_hot_actions = tf.cast(one_hot_actions, dtype=tf.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)
        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))
        return float(loss.numpy()), error
    
    def save(self, folder_name, **kwargs):
        """
        """
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')
        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')
        # Save info
        with open(folder_name + '/info.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information
    
    def load(self, folder_name, load_replay_buffer=True):
        """
        """
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')
        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer
        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')
        # Load info
        with open(folder_name + '/info.json', 'r') as f:
            info = json.load(f)
        if load_replay_buffer:
            self.replay_buffer.count = info['buff_count']
            self.replay_buffer.current = info['buff_curr']
        del info['buff_count'], info['buff_curr']  # just needed for the replay_buffer
        return info

config = dict (
  learning_rate = 0.00025,
  batch_size = 32,
  architecture = "DDDQN",
  infra = "Ubuntu"
)

pong_wrapper = PongWrapper(env, NO_OP_STEPS)
print("The environment has the following {} actions: {}".format(pong_wrapper.env.action_space.n, pong_wrapper.env.unwrapped.get_action_meanings()))
MAIN_DQN = build_q_network(pong_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(pong_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)
replay_buffer = ReplayBuffer(size=MEMORY_SIZE, input_shape=INPUT_SHAPE)
dddqn_agent = DDDQN_AGENT(MAIN_DQN, TARGET_DQN, replay_buffer, pong_wrapper.env.action_space.n, 
                    input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
                   replay_buffer_start_size=REPLAY_MEMORY_START_SIZE,
                   max_frames=MAX_FRAMES)

if PATH_LOAD_MODEL is not None:
    start_time = time.time()
    print('Loading model and info from the folder ', PATH_LOAD_MODEL)
    info = dddqn_agent.load(PATH_LOAD_MODEL, LOAD_REPLAY_BUFFER)

    # Apply information loaded from meta
    frame_number = info['frame_number']
    rewards = info['rewards']
    loss_list = info['loss_list']

    print(f'Loaded in {time.time() - start_time:.1f} seconds')
else:
    frame_number = 0
    rewards = []
    loss_list = []


def main():
    global frame_number, rewards, loss_list
    while frame_number < MAX_FRAMES:
        epoch_frame = 0
        print("frame_number: ", frame_number, ", MAX_FRAMES: ", MAX_FRAMES)
        while epoch_frame < EVAL_FREQUENCY:
            print("epoch frame: ", epoch_frame, " EVAL FREQ: ", EVAL_FREQUENCY)
            start_time = time.time()
            pong_wrapper.reset()
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):
                action = dddqn_agent.get_action(frame_number, pong_wrapper.state)
                processed_frame, reward, terminal, _ = pong_wrapper.step(action)
                frame_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                # Add experience to replay memory
                dddqn_agent.add_experience(action=action,
                                     frame=processed_frame[:, :, 0], # shape 84x84, remove last dimension
                                     reward=reward, clip_reward=CLIP_REWARD,
                                     terminal=terminal)
                # Update agent
                if frame_number % UPDATE_FREQ == 0 and dddqn_agent.replay_buffer.count > REPLAY_MEMORY_START_SIZE:
                    loss, _ = dddqn_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR)
                    loss_list.append(loss)
                # Update target network
                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                    dddqn_agent.update_target_network()
                # Break the loop when the game is over
                if terminal:
                    terminal = False
                    break
            rewards.append(episode_reward_sum)
            # wandb.log({'Game number': len(rewards), '# Frame': frame_number, '% Frame': round(frame_number / MAX_FRAMES, 2), "Average reward": round(np.mean(rewards[-10:]), 2), \
            #          "Time taken": round(time.time() - start_time, 2)})
        # Evaluation
        terminal = True
        eval_rewards = []
        evaluate_frame_number = 0
        for _ in range(EVAL_STEPS):
            if terminal:
                pong_wrapper.reset(evaluation=True)
                life_lost = True
                episode_reward_sum = 0
                terminal = False
            action = dddqn_agent.get_action(frame_number, pong_wrapper.state, evaluation=True)
            # Step action
            _, reward, terminal, truncated, info = pong_wrapper.step(action)
            evaluate_frame_number += 1
            episode_reward_sum += reward
            # On game-over
            if terminal:
                eval_rewards.append(episode_reward_sum)
        if len(eval_rewards) > 0:
            final_score = np.mean(eval_rewards)
        else:
            # In case the first game is longer than EVAL_STEPS
            final_score = episode_reward_sum
        # Log evaluation score
        # wandb.log({'# Frame': frame_number, '% Frame': round(frame_number / MAX_FRAMES, 2), 'Evaluation score': final_score})
        # Save the networks, frame number, rewards and losses. 
        if len(rewards) > 500 and PATH_SAVE_MODEL is not None:
            dddqn_agent.save(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(frame_number).zfill(8)}', \
                             frame_number=frame_number, rewards=rewards, loss_list=loss_list)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Save the model, I need this to save the networks, frame number, rewards and losses. 
        # if I want to stop the script and restart without training from the beginning
        if PATH_SAVE_MODEL is None:
            print("Setting path to ../model")
            PATH_SAVE_MODEL = "../model"
        print('Saving the model...')
        dddqn_agent.save(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(frame_number).zfill(8)}', \
                             frame_number=frame_number, rewards=rewards, loss_list=loss_list)
        print('Saved.')

# generate_random_game()