#!/usr/bin/env python3
"""
Intelligent Task Distribution System

Automatically assigns tasks to optimal agents/workers based on:
- Task type and complexity
- Agent capabilities and load
- Resource availability
- Historical performance

Inspired by HumanLayer's intelligent work distribution across agents/workers.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AgentCapability(Enum):
    """Agent capabilities"""
    RESEARCH = "research"
    PLANNING = "planning"
    CODING = "coding"
    TESTING = "testing"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class AgentProfile:
    """Profile of an available agent"""
    agent_id: str
    agent_type: str  # 'research', 'planner', 'implementer', 'verifier', 'general'
    capabilities: List[AgentCapability]
    max_complexity: TaskComplexity
    current_load: int = 0  # Number of active tasks
    max_load: int = 3
    success_rate: float = 1.0  # Historical success rate
    avg_completion_time: float = 0.0  # Average time in seconds
    specializations: List[str] = field(default_factory=list)

    def can_handle(self, capability: AgentCapability, complexity: TaskComplexity) -> bool:
        """Check if agent can handle a task"""
        if self.current_load >= self.max_load:
            return False

        if capability not in self.capabilities:
            return False

        complexity_order = {
            TaskComplexity.SIMPLE: 0,
            TaskComplexity.MEDIUM: 1,
            TaskComplexity.COMPLEX: 2,
            TaskComplexity.VERY_COMPLEX: 3
        }

        return complexity_order[complexity] <= complexity_order[self.max_complexity]

    def fitness_score(self, capability: AgentCapability, complexity: TaskComplexity) -> float:
        """
        Calculate fitness score for a task (0.0 - 1.0)

        Higher score = better fit
        """
        if not self.can_handle(capability, complexity):
            return 0.0

        score = 1.0

        # Prefer agents with lower load
        load_factor = 1.0 - (self.current_load / max(self.max_load, 1))
        score *= (0.5 + 0.5 * load_factor)

        # Consider success rate
        score *= (0.3 + 0.7 * self.success_rate)

        # Penalty for overqualified agents (save them for harder tasks)
        complexity_order = {
            TaskComplexity.SIMPLE: 0,
            TaskComplexity.MEDIUM: 1,
            TaskComplexity.COMPLEX: 2,
            TaskComplexity.VERY_COMPLEX: 3
        }
        if complexity_order[self.max_complexity] > complexity_order[complexity] + 1:
            score *= 0.8  # Slight penalty

        return min(score, 1.0)


@dataclass
class TaskRequest:
    """Task to be distributed"""
    task_id: str
    description: str
    required_capability: AgentCapability
    complexity: TaskComplexity
    priority: int = 5  # 1-10
    keywords: List[str] = field(default_factory=list)
    constraints: Dict = field(default_factory=dict)
    deadline: Optional[datetime] = None


@dataclass
class TaskAssignment:
    """Assignment of task to agent"""
    task_id: str
    agent_id: str
    confidence: float  # 0.0 - 1.0
    estimated_time: float
    assigned_at: datetime = field(default_factory=datetime.now)


class TaskDistributor:
    """
    Intelligent Task Distribution System

    Assigns tasks to optimal agents based on capabilities, load, and performance.
    """

    def __init__(self):
        """Initialize task distributor"""
        self.agents: Dict[str, AgentProfile] = {}
        self.assignments: Dict[str, TaskAssignment] = {}
        self.task_history: List[Dict] = []

        # Initialize default agent profiles
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialize default agent profiles"""

        # Research agent
        self.register_agent(AgentProfile(
            agent_id="research_1",
            agent_type="research",
            capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
            max_complexity=TaskComplexity.COMPLEX,
            max_load=5,
            specializations=["file_search", "codebase_analysis", "pattern_detection"]
        ))

        # Planner agent
        self.register_agent(AgentProfile(
            agent_id="planner_1",
            agent_type="planner",
            capabilities=[AgentCapability.PLANNING, AgentCapability.ANALYSIS],
            max_complexity=TaskComplexity.VERY_COMPLEX,
            max_load=3,
            specializations=["architecture", "implementation_planning", "task_breakdown"]
        ))

        # Implementer agents (multiple for parallel execution)
        for i in range(1, 4):
            self.register_agent(AgentProfile(
                agent_id=f"implementer_{i}",
                agent_type="implementer",
                capabilities=[AgentCapability.CODING],
                max_complexity=TaskComplexity.VERY_COMPLEX,
                max_load=2,
                specializations=["code_generation", "refactoring", "bug_fixing"]
            ))

        # Verifier agent
        self.register_agent(AgentProfile(
            agent_id="verifier_1",
            agent_type="verifier",
            capabilities=[AgentCapability.TESTING, AgentCapability.ANALYSIS],
            max_complexity=TaskComplexity.COMPLEX,
            max_load=5,
            specializations=["testing", "validation", "quality_assurance"]
        ))

        # Summarizer agents
        for i in range(1, 3):
            self.register_agent(AgentProfile(
                agent_id=f"summarizer_{i}",
                agent_type="summarizer",
                capabilities=[AgentCapability.SUMMARIZATION],
                max_complexity=TaskComplexity.MEDIUM,
                max_load=10,
                specializations=["compression", "summarization"]
            ))

        # General purpose agent
        self.register_agent(AgentProfile(
            agent_id="general_1",
            agent_type="general",
            capabilities=[
                AgentCapability.GENERAL,
                AgentCapability.ANALYSIS,
                AgentCapability.RESEARCH
            ],
            max_complexity=TaskComplexity.MEDIUM,
            max_load=5,
            specializations=["general_queries"]
        ))

    def register_agent(self, agent: AgentProfile):
        """Register an agent"""
        self.agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def infer_task_type(self, description: str) -> Tuple[AgentCapability, TaskComplexity]:
        """
        Infer task type and complexity from description

        Args:
            description: Task description

        Returns:
            (capability, complexity)
        """
        desc_lower = description.lower()

        # Capability detection
        capability = AgentCapability.GENERAL

        if any(kw in desc_lower for kw in ['research', 'find', 'search', 'explore', 'analyze architecture']):
            capability = AgentCapability.RESEARCH
        elif any(kw in desc_lower for kw in ['plan', 'design', 'architect', 'structure']):
            capability = AgentCapability.PLANNING
        elif any(kw in desc_lower for kw in ['implement', 'code', 'write', 'create', 'build', 'develop']):
            capability = AgentCapability.CODING
        elif any(kw in desc_lower for kw in ['test', 'verify', 'validate', 'check']):
            capability = AgentCapability.TESTING
        elif any(kw in desc_lower for kw in ['summarize', 'compress', 'condense']):
            capability = AgentCapability.SUMMARIZATION
        elif any(kw in desc_lower for kw in ['analyze', 'review', 'examine']):
            capability = AgentCapability.ANALYSIS

        # Complexity detection
        complexity = TaskComplexity.MEDIUM

        # Simple indicators
        if any(kw in desc_lower for kw in ['simple', 'quick', 'basic', 'easy', 'trivial']):
            complexity = TaskComplexity.SIMPLE
        # Complex indicators
        elif any(kw in desc_lower for kw in ['complex', 'advanced', 'sophisticated', 'intricate']):
            complexity = TaskComplexity.COMPLEX
        # Very complex indicators
        elif any(kw in desc_lower for kw in ['entire', 'full', 'complete', 'system-wide', 'refactor all']):
            complexity = TaskComplexity.VERY_COMPLEX
        # Estimate by length and technical terms
        else:
            word_count = len(description.split())
            technical_terms = len(re.findall(r'\b(api|database|authentication|authorization|integration|framework|architecture|microservice)\b', desc_lower))

            if word_count < 10 and technical_terms == 0:
                complexity = TaskComplexity.SIMPLE
            elif word_count > 30 or technical_terms > 3:
                complexity = TaskComplexity.COMPLEX

        return capability, complexity

    def assign_task(self, task: TaskRequest) -> Optional[TaskAssignment]:
        """
        Assign a task to the best available agent

        Args:
            task: Task to assign

        Returns:
            TaskAssignment if successful, None if no suitable agent
        """
        # Find all capable agents
        capable_agents = [
            (agent_id, agent)
            for agent_id, agent in self.agents.items()
            if agent.can_handle(task.required_capability, task.complexity)
        ]

        if not capable_agents:
            return None

        # Score each agent
        scored_agents = [
            (agent_id, agent, agent.fitness_score(task.required_capability, task.complexity))
            for agent_id, agent in capable_agents
        ]

        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[2], reverse=True)

        # Assign to best agent
        agent_id, agent, confidence = scored_agents[0]

        # Update agent load
        agent.current_load += 1

        # Estimate completion time
        if agent.avg_completion_time > 0:
            estimated_time = agent.avg_completion_time
        else:
            # Default estimates by complexity
            time_estimates = {
                TaskComplexity.SIMPLE: 30,
                TaskComplexity.MEDIUM: 120,
                TaskComplexity.COMPLEX: 300,
                TaskComplexity.VERY_COMPLEX: 600
            }
            estimated_time = time_estimates.get(task.complexity, 120)

        # Create assignment
        assignment = TaskAssignment(
            task_id=task.task_id,
            agent_id=agent_id,
            confidence=confidence,
            estimated_time=estimated_time
        )

        self.assignments[task.task_id] = assignment

        return assignment

    def complete_task(self, task_id: str, success: bool, actual_time: float):
        """
        Mark task as complete and update agent statistics

        Args:
            task_id: Task ID
            success: Whether task completed successfully
            actual_time: Actual completion time in seconds
        """
        if task_id not in self.assignments:
            return

        assignment = self.assignments[task_id]
        agent = self.agents.get(assignment.agent_id)

        if not agent:
            return

        # Update agent load
        agent.current_load = max(0, agent.current_load - 1)

        # Update success rate (exponential moving average)
        alpha = 0.1
        agent.success_rate = (1 - alpha) * agent.success_rate + alpha * (1.0 if success else 0.0)

        # Update average completion time
        if agent.avg_completion_time == 0:
            agent.avg_completion_time = actual_time
        else:
            agent.avg_completion_time = (1 - alpha) * agent.avg_completion_time + alpha * actual_time

        # Record in history
        self.task_history.append({
            "task_id": task_id,
            "agent_id": assignment.agent_id,
            "success": success,
            "actual_time": actual_time,
            "estimated_time": assignment.estimated_time,
            "completed_at": datetime.now().isoformat()
        })

    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]

        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "capabilities": [c.value for c in agent.capabilities],
            "current_load": agent.current_load,
            "max_load": agent.max_load,
            "utilization": agent.current_load / agent.max_load if agent.max_load > 0 else 0,
            "success_rate": agent.success_rate,
            "avg_completion_time": agent.avg_completion_time,
            "specializations": agent.specializations
        }

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        total_agents = len(self.agents)
        busy_agents = sum(1 for a in self.agents.values() if a.current_load > 0)
        total_capacity = sum(a.max_load for a in self.agents.values())
        current_load = sum(a.current_load for a in self.agents.values())

        # Group by type
        by_type = {}
        for agent in self.agents.values():
            if agent.agent_type not in by_type:
                by_type[agent.agent_type] = {
                    "count": 0,
                    "total_capacity": 0,
                    "current_load": 0
                }
            by_type[agent.agent_type]["count"] += 1
            by_type[agent.agent_type]["total_capacity"] += agent.max_load
            by_type[agent.agent_type]["current_load"] += agent.current_load

        return {
            "total_agents": total_agents,
            "busy_agents": busy_agents,
            "idle_agents": total_agents - busy_agents,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "utilization": current_load / total_capacity if total_capacity > 0 else 0,
            "by_type": by_type,
            "active_assignments": len(self.assignments),
            "completed_tasks": len(self.task_history)
        }

    def get_recommendations(self, description: str) -> Dict:
        """
        Get task assignment recommendations

        Args:
            description: Task description

        Returns:
            Recommendations with best agents
        """
        capability, complexity = self.infer_task_type(description)

        # Find capable agents and score them
        recommendations = []

        for agent_id, agent in self.agents.items():
            if agent.can_handle(capability, complexity):
                score = agent.fitness_score(capability, complexity)
                recommendations.append({
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type,
                    "confidence": score,
                    "current_load": agent.current_load,
                    "max_load": agent.max_load,
                    "success_rate": agent.success_rate
                })

        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "inferred_capability": capability.value,
            "inferred_complexity": complexity.value,
            "recommendations": recommendations[:5],  # Top 5
            "available_agents": len(recommendations)
        }


# Example usage
if __name__ == "__main__":
    print("Intelligent Task Distribution System")
    print("=" * 70)

    distributor = TaskDistributor()

    # Show system status
    status = distributor.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Agents: {status['total_agents']} ({status['idle_agents']} idle)")
    print(f"   Utilization: {status['utilization']:.1%}")
    print(f"   Capacity: {status['current_load']}/{status['total_capacity']}")

    # Test task assignment
    print(f"\n\n📋 Test Task Assignment:")

    test_tasks = [
        "Research authentication patterns in the codebase",
        "Implement rate limiting for API endpoints",
        "Write unit tests for the user service",
        "Summarize this 10,000 word document"
    ]

    for desc in test_tasks:
        print(f"\n   Task: {desc}")

        # Get recommendations
        rec = distributor.get_recommendations(desc)
        print(f"   Type: {rec['inferred_capability']} ({rec['inferred_complexity']})")

        if rec['recommendations']:
            best = rec['recommendations'][0]
            print(f"   Best agent: {best['agent_id']} (confidence: {best['confidence']:.2f})")
