#!/usr/bin/env python3
"""
MDP Policy Agent - Markov Decision Process for Control Flow

Makes execution decisions based on current state using policy-based approach.

Traditional: Fixed control flow (always retrieve → generate)
MDP-based: Dynamic decisions based on state (retrieve_more vs synthesize vs revise)

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Available actions in action space"""
    RETRIEVE_MORE = "retrieve_more"
    REFINE = "refine"
    SYNTHESIZE = "synthesize"
    REVISE_QUERY = "revise_query"
    CONTINUE = "continue"


@dataclass
class State:
    """MDP state representation"""
    iteration: int
    num_documents: int
    avg_relevance: float
    has_reflection: bool
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "num_documents": self.num_documents,
            "avg_relevance": self.avg_relevance,
            "has_reflection": self.has_reflection,
            "confidence": self.confidence
        }


class MDPPolicyAgent:
    """
    Policy-based decision making using MDP

    Usage:
        agent = MDPPolicyAgent()
        state = State(iteration=2, num_documents=5, avg_relevance=0.8, ...)
        action = agent.choose_action(state)
    """

    def __init__(self):
        """Initialize MDP policy agent"""
        # Q-values for (state_signature, action) pairs
        # In production, would use neural network
        self.q_values = {}

        # Policy parameters
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy
        self.discount_factor = 0.9   # Gamma for future rewards

    def choose_action(self, state: State) -> Action:
        """
        Choose action based on current state

        Args:
            state: Current execution state

        Returns:
            Chosen action
        """
        # Get state signature for Q-value lookup
        state_sig = self._get_state_signature(state)

        # Epsilon-greedy policy
        import random
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.choice(list(Action))
        else:
            # Exploit: best known action
            return self._get_best_action(state_sig, state)

    def _get_state_signature(self, state: State) -> str:
        """Get discrete state signature for Q-value lookup"""
        # Discretize continuous state into bins
        iteration_bin = "early" if state.iteration < 3 else "mid" if state.iteration < 7 else "late"
        docs_bin = "few" if state.num_documents < 3 else "some" if state.num_documents < 10 else "many"
        relevance_bin = "low" if state.avg_relevance < 0.5 else "med" if state.avg_relevance < 0.8 else "high"

        return f"{iteration_bin}_{docs_bin}_{relevance_bin}"

    def _get_best_action(self, state_sig: str, state: State) -> Action:
        """Get best action for state based on Q-values"""
        # Get Q-values for all actions in this state
        action_values = {}

        for action in Action:
            q_key = (state_sig, action)
            action_values[action] = self.q_values.get(q_key, 0.0)

        # If no Q-values yet, use heuristic policy
        if all(v == 0.0 for v in action_values.values()):
            return self._heuristic_policy(state)

        # Return action with highest Q-value
        return max(action_values, key=action_values.get)

    def _heuristic_policy(self, state: State) -> Action:
        """
        Heuristic policy for bootstrap (before learning)

        Rules:
        - If low docs and early: retrieve more
        - If good relevance and late: synthesize
        - If low relevance and mid: refine
        - Otherwise: continue
        """
        if state.num_documents < 3 and state.iteration < 5:
            return Action.RETRIEVE_MORE

        if state.avg_relevance > 0.7 and state.iteration > 3:
            return Action.SYNTHESIZE

        if state.avg_relevance < 0.5 and state.iteration < 7:
            return Action.REFINE

        return Action.CONTINUE

    def update_q_value(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State
    ):
        """
        Update Q-value based on observed reward (Q-learning)

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        state_sig = self._get_state_signature(state)
        next_state_sig = self._get_state_signature(next_state)

        # Current Q-value
        q_key = (state_sig, action)
        current_q = self.q_values.get(q_key, 0.0)

        # Max Q-value for next state
        next_q_values = [
            self.q_values.get((next_state_sig, a), 0.0)
            for a in Action
        ]
        max_next_q = max(next_q_values) if next_q_values else 0.0

        # Q-learning update
        learning_rate = 0.1
        new_q = current_q + learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_values[q_key] = new_q

    def get_policy_statistics(self) -> Dict:
        """Get policy learning statistics"""
        return {
            "learned_state_action_pairs": len(self.q_values),
            "exploration_rate": self.exploration_rate,
            "avg_q_value": sum(self.q_values.values()) / len(self.q_values) if self.q_values else 0.0
        }


if __name__ == "__main__":
    print("="*70)
    print("MDP Policy Agent Demo")
    print("="*70)

    agent = MDPPolicyAgent()

    # Test different states
    test_states = [
        State(iteration=1, num_documents=2, avg_relevance=0.4, has_reflection=False, confidence=0.3),
        State(iteration=5, num_documents=8, avg_relevance=0.85, has_reflection=True, confidence=0.9),
        State(iteration=8, num_documents=3, avg_relevance=0.6, has_reflection=True, confidence=0.5)
    ]

    print("\nHeuristic Policy (before learning):")
    print("-"*70)

    for i, state in enumerate(test_states, 1):
        action = agent.choose_action(state)
        print(f"\nState {i}:")
        print(f"  Iteration: {state.iteration}, Docs: {state.num_documents}, Relevance: {state.avg_relevance:.2f}")
        print(f"  Action: {action.value}")

    # Simulate learning
    print("\n" + "="*70)
    print("Simulating Q-learning...")
    print("-"*70)

    for _ in range(10):
        state = test_states[0]
        action = agent.choose_action(state)

        # Simulate reward (higher for good actions)
        reward = 1.0 if action == Action.RETRIEVE_MORE else 0.5

        # Simulate next state
        next_state = State(
            iteration=state.iteration + 1,
            num_documents=state.num_documents + 3,
            avg_relevance=0.7,
            has_reflection=True,
            confidence=0.7
        )

        agent.update_q_value(state, action, reward, next_state)

    stats = agent.get_policy_statistics()
    print(f"\nLearning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("MDP Policy Agent ready for integration")
    print("="*70)
