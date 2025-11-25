#!/usr/bin/env python3
"""
Supervisor Agent - Dynamic Task Routing and Strategy Selection

Routes tasks to optimal agents and selects best execution strategies dynamically.

Inspired by Deep-Thinking RAG supervisor pattern.

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Task type classification"""
    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VERIFICATION = "verification"


@dataclass
class RoutingDecision:
    """Agent routing decision"""
    agent_type: str
    strategy: str
    confidence: float
    reasoning: str


class SupervisorAgent:
    """
    Supervisor for dynamic task routing

    Usage:
        supervisor = SupervisorAgent()
        decision = supervisor.route_task("Analyze query performance")
    """

    def __init__(self):
        """Initialize supervisor"""
        # Available strategies per task type
        self.strategies = {
            TaskType.SEARCH: ["vector", "keyword", "hybrid"],
            TaskType.ANALYSIS: ["sequential", "parallel", "hierarchical"],
            TaskType.GENERATION: ["one-shot", "iterative", "chain-of-thought"]
        }

        # Strategy performance history (for learning)
        self.performance_history = {}

    def route_task(self, task: str, context: Dict = None) -> RoutingDecision:
        """
        Route task to best agent with optimal strategy

        Args:
            task: Task description
            context: Additional context

        Returns:
            RoutingDecision with agent and strategy
        """
        # Classify task type
        task_type = self._classify_task(task)

        # Select best strategy based on history
        strategy = self._select_strategy(task_type, context)

        # Determine agent type
        agent_type = self._select_agent(task_type, task)

        # Calculate confidence
        confidence = self._calculate_confidence(task_type, strategy)

        return RoutingDecision(
            agent_type=agent_type,
            strategy=strategy,
            confidence=confidence,
            reasoning=f"Task type: {task_type.value}, selected {strategy} strategy"
        )

    def _classify_task(self, task: str) -> TaskType:
        """Classify task type"""
        task_lower = task.lower()

        if any(keyword in task_lower for keyword in ["search", "find", "retrieve"]):
            return TaskType.SEARCH
        elif any(keyword in task_lower for keyword in ["analyze", "evaluate", "assess"]):
            return TaskType.ANALYSIS
        elif any(keyword in task_lower for keyword in ["generate", "create", "write"]):
            return TaskType.GENERATION
        elif any(keyword in task_lower for keyword in ["verify", "check", "validate"]):
            return TaskType.VERIFICATION
        else:
            return TaskType.ANALYSIS  # Default

    def _select_strategy(self, task_type: TaskType, context: Dict = None) -> str:
        """
        Select best strategy based on historical performance

        Args:
            task_type: Classified task type
            context: Additional context

        Returns:
            Strategy name
        """
        available_strategies = self.strategies.get(task_type, ["default"])

        # Check performance history
        if task_type in self.performance_history:
            # Select best performing strategy
            best_strategy = max(
                self.performance_history[task_type].items(),
                key=lambda x: x[1]["success_rate"]
            )[0]
            return best_strategy

        # Default strategy per task type
        defaults = {
            TaskType.SEARCH: "hybrid",
            TaskType.ANALYSIS: "sequential",
            TaskType.GENERATION: "iterative",
            TaskType.VERIFICATION: "sequential"
        }

        return defaults.get(task_type, available_strategies[0])

    def _select_agent(self, task_type: TaskType, task: str) -> str:
        """Select appropriate agent type"""
        # Map task types to agents
        agent_map = {
            TaskType.SEARCH: "retrieval_agent",
            TaskType.ANALYSIS: "analysis_agent",
            TaskType.GENERATION: "generation_agent",
            TaskType.VERIFICATION: "verification_agent"
        }

        return agent_map.get(task_type, "general_agent")

    def _calculate_confidence(self, task_type: TaskType, strategy: str) -> float:
        """Calculate confidence in routing decision"""
        # Base confidence
        confidence = 0.7

        # Increase if we have performance history
        if task_type in self.performance_history:
            if strategy in self.performance_history[task_type]:
                history = self.performance_history[task_type][strategy]
                confidence = history.get("success_rate", 0.7)

        return confidence

    def update_performance(
        self,
        task_type: TaskType,
        strategy: str,
        success: bool
    ):
        """
        Update strategy performance history

        Args:
            task_type: Task type
            strategy: Strategy used
            success: Whether task succeeded
        """
        if task_type not in self.performance_history:
            self.performance_history[task_type] = {}

        if strategy not in self.performance_history[task_type]:
            self.performance_history[task_type][strategy] = {
                "total": 0,
                "successes": 0,
                "success_rate": 0.5
            }

        history = self.performance_history[task_type][strategy]
        history["total"] += 1
        if success:
            history["successes"] += 1
        history["success_rate"] = history["successes"] / history["total"]


if __name__ == "__main__":
    print("="*70)
    print("Supervisor Agent Demo")
    print("="*70)

    supervisor = SupervisorAgent()

    test_tasks = [
        "Search for PostgreSQL optimization techniques",
        "Analyze query performance bottlenecks",
        "Generate SQL optimization report",
        "Verify database indexes are optimal"
    ]

    for task in test_tasks:
        decision = supervisor.route_task(task)
        print(f"\nTask: {task}")
        print(f"  Agent: {decision.agent_type}")
        print(f"  Strategy: {decision.strategy}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")

        # Simulate updating performance
        supervisor.update_performance(
            TaskType.SEARCH if "search" in task.lower() else TaskType.ANALYSIS,
            decision.strategy,
            success=True
        )

    print("\n" + "="*70)
    print("Performance History:")
    for task_type, strategies in supervisor.performance_history.items():
        print(f"\n{task_type.value}:")
        for strategy, history in strategies.items():
            print(f"  {strategy}: {history['success_rate']:.2%} success ({history['total']} attempts)")
