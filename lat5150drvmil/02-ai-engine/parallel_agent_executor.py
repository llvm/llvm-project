#!/usr/bin/env python3
"""
Parallel Agent Executor - "M U L T I C L A U D E" Implementation

Enables running multiple ACE-FCA workflows, subagents, or AI tasks simultaneously.
Inspired by HumanLayer's parallel Claude Code sessions.

Key Features:
- Concurrent workflow execution with isolated contexts
- Async task management with asyncio
- Progress tracking across all parallel agents
- Resource management (max concurrent agents)
- Results aggregation and reporting

Based on: HumanLayer/CodeLayer parallel execution patterns
"""

import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AgentStatus(Enum):
    """Status of parallel agent execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ParallelTask:
    """Represents a task to be executed in parallel"""
    task_id: str
    task_type: str  # 'workflow', 'subagent', 'query'
    description: str
    params: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more important
    status: AgentStatus = AgentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 - 1.0
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "agent_id": self.agent_id,
            "has_error": bool(self.error)
        }

    def duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


class ParallelAgentExecutor:
    """
    Parallel Agent Executor - Run multiple AI agents simultaneously

    Implements "M U L T I C L A U D E" pattern from HumanLayer:
    - Concurrent workflow execution
    - Resource management (max parallel agents)
    - Progress tracking
    - Results aggregation
    """

    def __init__(self,
                 orchestrator,  # UnifiedAIOrchestrator instance
                 max_concurrent_agents: int = 3,
                 enable_progress_tracking: bool = True):
        """
        Initialize parallel agent executor

        Args:
            orchestrator: UnifiedAIOrchestrator with ACE-FCA enabled
            max_concurrent_agents: Maximum number of agents to run simultaneously
            enable_progress_tracking: Enable real-time progress updates
        """
        self.orchestrator = orchestrator
        self.max_concurrent_agents = max_concurrent_agents
        self.enable_progress_tracking = enable_progress_tracking

        self.tasks: Dict[str, ParallelTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Dict] = {}

        self._task_counter = 0
        self._executor_running = False
        self._executor_task: Optional[asyncio.Task] = None

    def submit_workflow(self,
                       description: str,
                       task_type: str = "feature",
                       complexity: str = "medium",
                       constraints: List[str] = None,
                       priority: int = 5) -> str:
        """
        Submit an ACE-FCA workflow for parallel execution

        Args:
            description: Task description
            task_type: 'feature', 'bugfix', 'refactor', 'analysis'
            complexity: 'simple', 'medium', 'complex'
            constraints: List of constraints
            priority: Task priority (1-10)

        Returns:
            Task ID for tracking
        """
        task_id = f"wf_{self._task_counter}"
        self._task_counter += 1

        task = ParallelTask(
            task_id=task_id,
            task_type="workflow",
            description=description,
            params={
                "task_description": description,
                "task_type": task_type,
                "complexity": complexity,
                "constraints": constraints or [],
                "model_preference": "quality_code"
            },
            priority=priority
        )

        self.tasks[task_id] = task
        asyncio.create_task(self.task_queue.put(task))

        return task_id

    def submit_subagent(self,
                       agent_type: str,
                       task_params: Dict,
                       description: str = None,
                       priority: int = 5) -> str:
        """
        Submit a specialized subagent task for parallel execution

        Args:
            agent_type: 'research', 'planner', 'implementer', 'verifier', 'summarizer'
            task_params: Parameters for the subagent
            description: Task description
            priority: Task priority (1-10)

        Returns:
            Task ID for tracking
        """
        task_id = f"sub_{self._task_counter}"
        self._task_counter += 1

        task = ParallelTask(
            task_id=task_id,
            task_type="subagent",
            description=description or f"{agent_type} task",
            params={
                "agent_type": agent_type,
                "task_params": task_params
            },
            priority=priority
        )

        self.tasks[task_id] = task
        asyncio.create_task(self.task_queue.put(task))

        return task_id

    def submit_query(self,
                    query: str,
                    model: str = "quality_code",
                    priority: int = 5) -> str:
        """
        Submit a simple AI query for parallel execution

        Args:
            query: Query text
            model: Model to use
            priority: Task priority (1-10)

        Returns:
            Task ID for tracking
        """
        task_id = f"qry_{self._task_counter}"
        self._task_counter += 1

        task = ParallelTask(
            task_id=task_id,
            task_type="query",
            description=query[:100],
            params={
                "query": query,
                "model": model
            },
            priority=priority
        )

        self.tasks[task_id] = task
        asyncio.create_task(self.task_queue.put(task))

        return task_id

    async def start(self):
        """Start the parallel agent executor"""
        if self._executor_running:
            return

        self._executor_running = True
        self._executor_task = asyncio.create_task(self._executor_loop())
        print(f"🚀 Parallel Agent Executor started (max {self.max_concurrent_agents} concurrent agents)")

    async def stop(self):
        """Stop the parallel agent executor"""
        if not self._executor_running:
            return

        self._executor_running = False

        # Cancel all running tasks
        for task_id, async_task in list(self.running_tasks.items()):
            async_task.cancel()
            if task_id in self.tasks:
                self.tasks[task_id].status = AgentStatus.CANCELLED

        if self._executor_task:
            self._executor_task.cancel()

        print("🛑 Parallel Agent Executor stopped")

    async def _executor_loop(self):
        """Main executor loop - manages parallel task execution"""
        while self._executor_running:
            try:
                # Check if we can start more tasks
                if len(self.running_tasks) < self.max_concurrent_agents:
                    try:
                        # Get next task from queue (non-blocking with timeout)
                        task = await asyncio.wait_for(self.task_queue.get(), timeout=0.5)

                        # Start task execution
                        async_task = asyncio.create_task(self._execute_task(task))
                        self.running_tasks[task.task_id] = async_task

                    except asyncio.TimeoutError:
                        pass  # No tasks in queue

                # Clean up completed tasks
                completed_tasks = []
                for task_id, async_task in list(self.running_tasks.items()):
                    if async_task.done():
                        completed_tasks.append(task_id)

                for task_id in completed_tasks:
                    del self.running_tasks[task_id]

                await asyncio.sleep(0.1)  # Prevent busy loop

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Executor loop error: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: ParallelTask):
        """Execute a single task"""
        task.status = AgentStatus.RUNNING
        task.started_at = datetime.now()
        task.agent_id = f"agent_{len(self.running_tasks)}"

        print(f"▶️  [{task.agent_id}] Starting: {task.description}")

        try:
            # Execute based on task type
            if task.task_type == "workflow":
                result = await self._execute_workflow(task)
            elif task.task_type == "subagent":
                result = await self._execute_subagent(task)
            elif task.task_type == "query":
                result = await self._execute_query(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            task.result = result
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 1.0

            self.results[task.task_id] = result

            duration = task.duration()
            print(f"✅ [{task.agent_id}] Completed: {task.description} ({duration:.1f}s)")

        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            print(f"❌ [{task.agent_id}] Failed: {task.description} - {str(e)}")

    async def _execute_workflow(self, task: ParallelTask) -> Dict:
        """Execute ACE-FCA workflow in parallel"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def run_workflow():
            return self.orchestrator.execute_workflow(**task.params)

        result = await loop.run_in_executor(None, run_workflow)
        return result

    async def _execute_subagent(self, task: ParallelTask) -> Dict:
        """Execute specialized subagent in parallel"""
        loop = asyncio.get_event_loop()

        def run_subagent():
            return self.orchestrator.use_subagent(
                task.params["agent_type"],
                task.params["task_params"]
            )

        result = await loop.run_in_executor(None, run_subagent)
        return result

    async def _execute_query(self, task: ParallelTask) -> Dict:
        """Execute simple query in parallel"""
        loop = asyncio.get_event_loop()

        def run_query():
            return self.orchestrator.local.generate(
                task.params["query"],
                model_selection=task.params["model"]
            )

        result = await loop.run_in_executor(None, run_query)
        return result

    def get_status(self) -> Dict:
        """Get comprehensive status of parallel executor"""
        pending = sum(1 for t in self.tasks.values() if t.status == AgentStatus.PENDING)
        running = sum(1 for t in self.tasks.values() if t.status == AgentStatus.RUNNING)
        completed = sum(1 for t in self.tasks.values() if t.status == AgentStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == AgentStatus.FAILED)

        return {
            "executor_running": self._executor_running,
            "max_concurrent_agents": self.max_concurrent_agents,
            "currently_running": len(self.running_tasks),
            "queue_size": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "tasks_by_status": {
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed
            },
            "running_tasks": [
                self.tasks[task_id].to_dict()
                for task_id in self.running_tasks.keys()
            ]
        }

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None

        return self.tasks[task_id].to_dict()

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get result of a completed task"""
        return self.results.get(task_id)

    def list_tasks(self, status: Optional[AgentStatus] = None) -> List[Dict]:
        """List all tasks, optionally filtered by status"""
        tasks = self.tasks.values()

        if status:
            tasks = [t for t in tasks if t.status == status]

        return [t.to_dict() for t in sorted(tasks, key=lambda t: t.created_at, reverse=True)]

    async def wait_for_completion(self, task_ids: List[str] = None, timeout: float = None) -> Dict:
        """
        Wait for tasks to complete

        Args:
            task_ids: Specific task IDs to wait for (None = all tasks)
            timeout: Timeout in seconds (None = no timeout)

        Returns:
            Dict with completion status
        """
        start_time = time.time()
        target_tasks = task_ids or list(self.tasks.keys())

        while True:
            # Check if all target tasks are done
            all_done = all(
                self.tasks[tid].status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]
                for tid in target_tasks
                if tid in self.tasks
            )

            if all_done:
                break

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {
                    "timed_out": True,
                    "elapsed": time.time() - start_time,
                    "completed_tasks": [
                        tid for tid in target_tasks
                        if tid in self.tasks and self.tasks[tid].status == AgentStatus.COMPLETED
                    ]
                }

            await asyncio.sleep(0.5)

        return {
            "timed_out": False,
            "elapsed": time.time() - start_time,
            "completed_tasks": [
                tid for tid in target_tasks
                if tid in self.tasks and self.tasks[tid].status == AgentStatus.COMPLETED
            ],
            "failed_tasks": [
                tid for tid in target_tasks
                if tid in self.tasks and self.tasks[tid].status == AgentStatus.FAILED
            ]
        }


# Example usage
if __name__ == "__main__":
    print("Parallel Agent Executor - 'M U L T I C L A U D E' Implementation")
    print("=" * 70)

    # Mock orchestrator for testing
    class MockOrchestrator:
        class local:
            @staticmethod
            def generate(query, model_selection="code"):
                time.sleep(2)  # Simulate work
                return {"response": f"Mock response for: {query[:50]}"}

        @staticmethod
        def execute_workflow(**kwargs):
            time.sleep(5)  # Simulate workflow
            return {"success": True, "phases_completed": ["research", "plan"]}

        @staticmethod
        def use_subagent(agent_type, params):
            time.sleep(3)  # Simulate subagent work
            return {"success": True, "compressed_output": "Mock subagent result"}

    async def test_parallel_execution():
        orchestrator = MockOrchestrator()
        executor = ParallelAgentExecutor(orchestrator, max_concurrent_agents=3)

        # Start executor
        await executor.start()

        # Submit multiple tasks
        tasks = []
        tasks.append(executor.submit_workflow("Add authentication to API", priority=10))
        tasks.append(executor.submit_workflow("Implement rate limiting", priority=8))
        tasks.append(executor.submit_subagent("research", {"query": "auth patterns"}, priority=7))
        tasks.append(executor.submit_query("How to optimize database queries?", priority=5))

        print(f"\n📋 Submitted {len(tasks)} tasks for parallel execution\n")

        # Show status
        status = executor.get_status()
        print(f"Status: {status['currently_running']}/{status['max_concurrent_agents']} running")
        print(f"Queue: {status['queue_size']} pending\n")

        # Wait for completion
        result = await executor.wait_for_completion(timeout=30)

        print(f"\n✅ Execution complete!")
        print(f"   Time: {result['elapsed']:.1f}s")
        print(f"   Completed: {len(result['completed_tasks'])}")
        print(f"   Failed: {len(result.get('failed_tasks', []))}")

        # Stop executor
        await executor.stop()

    # Run test
    asyncio.run(test_parallel_execution())
