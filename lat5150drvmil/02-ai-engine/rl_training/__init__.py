"""
Reinforcement Learning Training Pipeline

Implements RL training for self-improving agents:
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
- Reward functions
- Trajectory collection

Based on "Building a Training Architecture for Self-Improving AI Agents"
"""

from .reward_functions import RewardCalculator, TaskReward
from .trajectory_collector import TrajectoryCollector, Trajectory

__all__ = ['RewardCalculator', 'TaskReward', 'TrajectoryCollector', 'Trajectory']
