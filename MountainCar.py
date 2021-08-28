import gym

env = gym.make("MountainCar-v0")
env.reset()

done = False

while not done:
    action = action.action_space.sample()
    new_state, reward, done, _ = env.step(action)
    env.render()