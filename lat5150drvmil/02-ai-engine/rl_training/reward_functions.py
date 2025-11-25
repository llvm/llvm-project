#!/usr/bin/env python3
"""
Reward Functions for Reinforcement Learning

Defines reward signals for agent actions:
- Task completion rewards
- Quality-based rewards
- Efficiency rewards
- User feedback rewards

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of rewards"""
    TASK_SUCCESS = "task_success"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    USER_FEEDBACK = "user_feedback"
    INTERMEDIATE = "intermediate"


@dataclass
class TaskReward:
    """Reward for a task or action"""
    reward_type: str
    value: float
    reason: str
    metadata: Dict[str, Any] = None


class RewardCalculator:
    """
    Calculate rewards for agent actions

    Usage:
        calculator = RewardCalculator()
        reward = calculator.calculate_task_reward(
            success=True,
            quality_score=0.9,
            execution_time=10.5,
            user_rating=5
        )
    """

    def __init__(
        self,
        success_reward: float = 10.0,
        failure_penalty: float = -5.0,
        quality_weight: float = 5.0,
        efficiency_weight: float = 2.0,
        user_feedback_weight: float = 3.0
    ):
        """
        Initialize reward calculator

        Args:
            success_reward: Reward for task success
            failure_penalty: Penalty for task failure
            quality_weight: Weight for quality scores
            efficiency_weight: Weight for efficiency
            user_feedback_weight: Weight for user feedback
        """
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.quality_weight = quality_weight
        self.efficiency_weight = efficiency_weight
        self.user_feedback_weight = user_feedback_weight

    def calculate_task_reward(
        self,
        success: bool,
        quality_score: float = None,
        execution_time: float = None,
        expected_time: float = None,
        user_rating: int = None
    ) -> float:
        """
        Calculate total reward for task

        Args:
            success: Whether task succeeded
            quality_score: Quality score 0.0-1.0
            execution_time: Actual execution time
            expected_time: Expected execution time
            user_rating: User rating 1-5 stars

        Returns:
            Total reward value
        """
        total_reward = 0.0

        # Base reward for success/failure
        if success:
            total_reward += self.success_reward
        else:
            total_reward += self.failure_penalty

        # Quality reward (0.0-1.0 → 0.0-quality_weight)
        if quality_score is not None:
            quality_reward = quality_score * self.quality_weight
            total_reward += quality_reward

        # Efficiency reward (faster = better)
        if execution_time is not None and expected_time is not None:
            efficiency = self._calculate_efficiency_reward(
                execution_time,
                expected_time
            )
            total_reward += efficiency

        # User feedback reward (1-5 stars → -weight to +weight)
        if user_rating is not None:
            feedback_reward = self._calculate_feedback_reward(user_rating)
            total_reward += feedback_reward

        return total_reward

    def _calculate_efficiency_reward(
        self,
        execution_time: float,
        expected_time: float
    ) -> float:
        """
        Calculate efficiency reward

        Faster than expected: positive reward
        Slower than expected: negative reward
        """
        # Time ratio: <1 = faster, >1 = slower
        time_ratio = execution_time / expected_time

        if time_ratio < 1.0:
            # Faster than expected: reward
            speedup = 1.0 - time_ratio
            return speedup * self.efficiency_weight
        else:
            # Slower than expected: penalty
            slowdown = time_ratio - 1.0
            return -slowdown * self.efficiency_weight * 0.5  # Gentler penalty

    def _calculate_feedback_reward(self, rating: int) -> float:
        """
        Calculate user feedback reward

        1-2 stars: negative
        3 stars: neutral
        4-5 stars: positive
        """
        # Normalize to -1 to +1
        normalized = (rating - 3) / 2.0
        return normalized * self.user_feedback_weight

    def calculate_intermediate_reward(
        self,
        action: str,
        state_improvement: float
    ) -> float:
        """
        Calculate reward for intermediate actions

        Args:
            action: Action taken
            state_improvement: How much state improved (0.0-1.0)

        Returns:
            Reward value
        """
        # Reward intermediate progress
        base_reward = 1.0
        return base_reward * state_improvement

    def calculate_shaped_reward(
        self,
        current_state: Dict,
        next_state: Dict,
        action: str
    ) -> float:
        """
        Reward shaping based on state transition

        Encourages progress toward goal without waiting for final outcome.

        Args:
            current_state: State before action
            next_state: State after action
            action: Action taken

        Returns:
            Shaped reward
        """
        reward = 0.0

        # Reward for document retrieval
        if action == "retrieve":
            doc_increase = len(next_state.get("documents", [])) - len(current_state.get("documents", []))
            if doc_increase > 0:
                reward += doc_increase * 0.5

        # Reward for relevance improvement
        current_relevance = current_state.get("avg_relevance", 0.0)
        next_relevance = next_state.get("avg_relevance", 0.0)
        if next_relevance > current_relevance:
            reward += (next_relevance - current_relevance) * 2.0

        # Penalty for excessive iterations
        if next_state.get("iteration", 0) > 5:
            reward -= 0.1 * (next_state["iteration"] - 5)

        return reward


if __name__ == "__main__":
    print("="*70)
    print("Reward Functions Demo")
    print("="*70)

    calculator = RewardCalculator()

    # Test case 1: Successful task with good quality
    reward = calculator.calculate_task_reward(
        success=True,
        quality_score=0.9,
        execution_time=8.0,
        expected_time=10.0,
        user_rating=5
    )
    print(f"\nCase 1: Successful task with high quality")
    print(f"  Reward: {reward:.2f}")

    # Test case 2: Failed task
    reward = calculator.calculate_task_reward(
        success=False,
        quality_score=0.3,
        user_rating=2
    )
    print(f"\nCase 2: Failed task")
    print(f"  Reward: {reward:.2f}")

    # Test case 3: Successful but slow
    reward = calculator.calculate_task_reward(
        success=True,
        quality_score=0.7,
        execution_time=20.0,
        expected_time=10.0,
        user_rating=3
    )
    print(f"\nCase 3: Successful but slow")
    print(f"  Reward: {reward:.2f}")

    # Test case 4: Intermediate action
    reward = calculator.calculate_intermediate_reward(
        action="retrieve",
        state_improvement=0.6
    )
    print(f"\nCase 4: Intermediate action")
    print(f"  Reward: {reward:.2f}")

    # Test case 5: Reward shaping
    current_state = {"documents": [], "avg_relevance": 0.5, "iteration": 2}
    next_state = {"documents": [1, 2, 3], "avg_relevance": 0.8, "iteration": 3}
    reward = calculator.calculate_shaped_reward(current_state, next_state, "retrieve")
    print(f"\nCase 5: Reward shaping")
    print(f"  Reward: {reward:.2f}")
