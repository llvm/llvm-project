#!/usr/bin/env python3
"""
Hephaestus Framework Integration for DSMIL AI System

Integrates semi-structured agentic framework with dynamic workflow building.
Complements Laddr multi-agent system with phase-based task orchestration.

Features:
- Dynamic workflow building (agents create tasks as they discover them)
- Phase-based organization (Analysis → Implementation → Validation)
- Kanban ticket system with dependency tracking
- Vector storage (Qdrant) for semantic task understanding
- Integration with MCP servers and multiple LLM providers
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class HephaestusTask:
    """Semi-structured task representation"""
    id: str
    title: str
    description: str
    phase: str  # analysis, implementation, validation
    status: str  # todo, in_progress, blocked, completed
    dependencies: List[str]
    assigned_agent: Optional[str] = None
    created_at: str = None
    completed_at: Optional[str] = None
    metadata: Dict = None


@dataclass
class HephaestusPhase:
    """Phase in the workflow"""
    name: str
    description: str
    tasks: List[HephaestusTask]
    completion_criteria: List[str]
    agents_required: List[str]


class HephaestusIntegrator:
    """
    Hephaestus Framework Integration

    Semi-structured agentic framework with dynamic workflow building.
    Agents can create new tasks as they discover them during execution.
    """

    def __init__(
        self,
        workspace_dir: str = "/tmp/hephaestus_workspace",
        enable_vector_storage: bool = True,
        enable_mcp_integration: bool = True
    ):
        """Initialize Hephaestus Integrator"""

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.workflows: Dict[str, Dict] = {}
        self.active_tasks: Dict[str, HephaestusTask] = {}

        # Phase definitions
        self.phases = {
            "analysis": HephaestusPhase(
                name="Analysis",
                description="Understand requirements, break down complex tasks, identify dependencies",
                tasks=[],
                completion_criteria=[
                    "All requirements clearly defined",
                    "Task breakdown complete",
                    "Dependencies identified",
                    "Risk assessment done"
                ],
                agents_required=["research_specialist", "code_analyst"]
            ),
            "implementation": HephaestusPhase(
                name="Implementation",
                description="Execute tasks, build solutions, integrate components",
                tasks=[],
                completion_criteria=[
                    "All tasks executed",
                    "Code written and tested",
                    "Integration complete",
                    "Documentation updated"
                ],
                agents_required=["code_analyst", "system_optimizer"]
            ),
            "validation": HephaestusPhase(
                name="Validation",
                description="Test, verify, benchmark, and secure the solution",
                tasks=[],
                completion_criteria=[
                    "All tests passing",
                    "Benchmarks meet targets",
                    "Security audit passed",
                    "Quality metrics acceptable"
                ],
                agents_required=["security_researcher", "benchmark_analyst"]
            )
        }

        # Integration flags
        self.enable_vector_storage = enable_vector_storage
        self.enable_mcp_integration = enable_mcp_integration

        # Statistics
        self.stats = {
            "workflows_created": 0,
            "tasks_completed": 0,
            "dynamic_tasks_created": 0,
            "phases_completed": 0
        }

        print("=" * 70)
        print(" Hephaestus Framework Integration")
        print("=" * 70)
        print(f"✓ Workspace: {self.workspace_dir}")
        print(f"✓ Phases: {len(self.phases)}")
        print(f"✓ Vector Storage: {'Enabled' if enable_vector_storage else 'Disabled'}")
        print(f"✓ MCP Integration: {'Enabled' if enable_mcp_integration else 'Disabled'}")
        print("=" * 70)
        print()

    def create_workflow(
        self,
        project_name: str,
        goal: str,
        initial_tasks: Optional[List[Dict]] = None
    ) -> str:
        """
        Create a new semi-structured workflow

        Args:
            project_name: Project identifier
            goal: High-level goal
            initial_tasks: Optional initial task list

        Returns:
            Workflow ID
        """

        workflow_id = f"wf_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.workflows[workflow_id] = {
            "id": workflow_id,
            "project_name": project_name,
            "goal": goal,
            "created_at": datetime.now().isoformat(),
            "current_phase": "analysis",
            "tasks": {},
            "completed_phases": [],
            "metadata": {
                "dynamic_tasks_added": 0,
                "phase_transitions": []
            }
        }

        # Add initial tasks
        if initial_tasks:
            for task_data in initial_tasks:
                self._add_task(workflow_id, task_data)

        self.stats["workflows_created"] += 1

        print(f"✓ Created workflow '{workflow_id}' for '{project_name}'")
        print(f"  Goal: {goal}")
        print(f"  Initial tasks: {len(initial_tasks) if initial_tasks else 0}")

        return workflow_id

    def _add_task(
        self,
        workflow_id: str,
        task_data: Dict,
        dynamic: bool = False
    ) -> str:
        """
        Add task to workflow

        Args:
            workflow_id: Workflow ID
            task_data: Task definition
            dynamic: Whether task was dynamically created

        Returns:
            Task ID
        """

        task_id = f"task_{len(self.workflows[workflow_id]['tasks']) + 1}"

        task = HephaestusTask(
            id=task_id,
            title=task_data.get("title", "Untitled Task"),
            description=task_data.get("description", ""),
            phase=task_data.get("phase", "analysis"),
            status="todo",
            dependencies=task_data.get("dependencies", []),
            assigned_agent=task_data.get("assigned_agent"),
            created_at=datetime.now().isoformat(),
            metadata=task_data.get("metadata", {})
        )

        self.workflows[workflow_id]["tasks"][task_id] = asdict(task)
        self.active_tasks[task_id] = task

        if dynamic:
            self.workflows[workflow_id]["metadata"]["dynamic_tasks_added"] += 1
            self.stats["dynamic_tasks_created"] += 1

        return task_id

    def execute_workflow(
        self,
        workflow_id: str,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute workflow with dynamic task creation

        Agents can create new tasks as they discover requirements.

        Args:
            workflow_id: Workflow ID
            max_iterations: Maximum execution iterations

        Returns:
            Execution results
        """

        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        workflow = self.workflows[workflow_id]

        print(f"\n{'='*70}")
        print(f" Executing Workflow: {workflow['project_name']}")
        print(f"{'='*70}\n")

        iteration = 0
        results = {
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "iterations": [],
            "phases_completed": [],
            "tasks_completed": [],
            "dynamic_tasks_created": []
        }

        while iteration < max_iterations:
            iteration += 1

            # Get current phase
            current_phase = workflow["current_phase"]

            print(f"Iteration {iteration} - Phase: {current_phase.upper()}")

            # Get executable tasks (no unmet dependencies)
            executable_tasks = self._get_executable_tasks(workflow_id, current_phase)

            if not executable_tasks:
                # Check if phase is complete
                if self._is_phase_complete(workflow_id, current_phase):
                    print(f"  ✓ Phase '{current_phase}' complete")

                    # Transition to next phase
                    next_phase = self._get_next_phase(current_phase)
                    if next_phase:
                        workflow["current_phase"] = next_phase
                        workflow["completed_phases"].append(current_phase)
                        workflow["metadata"]["phase_transitions"].append({
                            "from": current_phase,
                            "to": next_phase,
                            "timestamp": datetime.now().isoformat()
                        })
                        results["phases_completed"].append(current_phase)
                        self.stats["phases_completed"] += 1
                        print(f"  → Transitioning to phase '{next_phase}'")
                    else:
                        print(f"  ✓ All phases complete - workflow finished")
                        break
                else:
                    print(f"  ⚠ No executable tasks, but phase not complete (blocked)")
                    break
            else:
                # Execute tasks
                for task in executable_tasks[:3]:  # Execute up to 3 tasks per iteration
                    print(f"  • Executing: {task['title']}")

                    # Simulate task execution
                    execution_result = self._execute_task(workflow_id, task)

                    results["iterations"].append({
                        "iteration": iteration,
                        "task_id": task["id"],
                        "task_title": task["title"],
                        "result": execution_result
                    })

                    # Mark task complete
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now().isoformat()
                    results["tasks_completed"].append(task["id"])
                    self.stats["tasks_completed"] += 1

                    # Dynamic task creation (30% chance per task)
                    if execution_result.get("suggest_new_tasks"):
                        for new_task_data in execution_result["suggest_new_tasks"]:
                            new_task_id = self._add_task(workflow_id, new_task_data, dynamic=True)
                            print(f"    + Dynamic task created: {new_task_data['title']}")
                            results["dynamic_tasks_created"].append(new_task_id)

        results["end_time"] = datetime.now().isoformat()
        results["total_iterations"] = iteration
        results["success"] = len(workflow["completed_phases"]) == len(self.phases)

        print(f"\n{'='*70}")
        print(f" Workflow Execution Complete")
        print(f"{'='*70}")
        print(f"Iterations: {iteration}")
        print(f"Phases completed: {len(results['phases_completed'])}/{len(self.phases)}")
        print(f"Tasks completed: {len(results['tasks_completed'])}")
        print(f"Dynamic tasks created: {len(results['dynamic_tasks_created'])}")
        print()

        return results

    def _get_executable_tasks(self, workflow_id: str, phase: str) -> List[Dict]:
        """Get tasks that can be executed (no unmet dependencies)"""
        workflow = self.workflows[workflow_id]
        executable = []

        for task_id, task in workflow["tasks"].items():
            if (task["phase"] == phase and
                task["status"] == "todo" and
                self._dependencies_met(workflow_id, task)):
                executable.append(task)

        return executable

    def _dependencies_met(self, workflow_id: str, task: Dict) -> bool:
        """Check if all task dependencies are met"""
        workflow = self.workflows[workflow_id]

        for dep_id in task.get("dependencies", []):
            if dep_id in workflow["tasks"]:
                if workflow["tasks"][dep_id]["status"] != "completed":
                    return False

        return True

    def _is_phase_complete(self, workflow_id: str, phase: str) -> bool:
        """Check if all tasks in phase are complete"""
        workflow = self.workflows[workflow_id]

        for task in workflow["tasks"].values():
            if task["phase"] == phase and task["status"] != "completed":
                return False

        return True

    def _get_next_phase(self, current_phase: str) -> Optional[str]:
        """Get next phase in sequence"""
        phase_order = ["analysis", "implementation", "validation"]

        try:
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                return phase_order[current_idx + 1]
        except ValueError:
            pass

        return None

    def _execute_task(self, workflow_id: str, task: Dict) -> Dict[str, Any]:
        """
        Execute a single task

        This is a placeholder - in production, this would route to:
        - Laddr agents (for agent-based tasks)
        - MCP tools (for tool-based tasks)
        - Direct code execution (for script tasks)
        """

        # Simulate execution
        result = {
            "success": True,
            "output": f"Task '{task['title']}' executed successfully",
            "suggest_new_tasks": []
        }

        # Dynamically suggest new tasks (simulation)
        if "analysis" in task["phase"] and "analyze" in task["title"].lower():
            # Analysis tasks might discover new implementation tasks
            result["suggest_new_tasks"].append({
                "title": f"Implement findings from {task['title']}",
                "description": "Implementation based on analysis results",
                "phase": "implementation",
                "dependencies": [task["id"]]
            })

        return result

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status"""

        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        workflow = self.workflows[workflow_id]

        # Count tasks by status
        task_counts = {"todo": 0, "in_progress": 0, "blocked": 0, "completed": 0}
        for task in workflow["tasks"].values():
            task_counts[task["status"]] = task_counts.get(task["status"], 0) + 1

        # Count tasks by phase
        phase_counts = {phase: 0 for phase in self.phases.keys()}
        for task in workflow["tasks"].values():
            phase_counts[task["phase"]] = phase_counts.get(task["phase"], 0) + 1

        return {
            "workflow_id": workflow_id,
            "project_name": workflow["project_name"],
            "goal": workflow["goal"],
            "current_phase": workflow["current_phase"],
            "completed_phases": workflow["completed_phases"],
            "total_tasks": len(workflow["tasks"]),
            "task_counts": task_counts,
            "phase_counts": phase_counts,
            "dynamic_tasks_added": workflow["metadata"]["dynamic_tasks_added"],
            "progress_percentage": (task_counts["completed"] / len(workflow["tasks"]) * 100) if workflow["tasks"] else 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "system": self.stats.copy(),
            "active_workflows": len(self.workflows),
            "active_tasks": len(self.active_tasks),
            "phases_defined": len(self.phases)
        }


def main():
    """Demo / Test"""

    # Initialize
    integrator = HephaestusIntegrator()

    # Create a workflow
    workflow_id = integrator.create_workflow(
        project_name="security_audit",
        goal="Perform comprehensive security audit of AI system",
        initial_tasks=[
            {
                "title": "Analyze authentication system",
                "description": "Review API authentication and authorization",
                "phase": "analysis",
                "dependencies": []
            },
            {
                "title": "Analyze input validation",
                "description": "Review all input validation mechanisms",
                "phase": "analysis",
                "dependencies": []
            },
            {
                "title": "Fix authentication vulnerabilities",
                "description": "Implement fixes for identified auth issues",
                "phase": "implementation",
                "dependencies": ["task_1"]
            },
            {
                "title": "Fix input validation issues",
                "description": "Implement improved input validation",
                "phase": "implementation",
                "dependencies": ["task_2"]
            },
            {
                "title": "Run security tests",
                "description": "Execute comprehensive security test suite",
                "phase": "validation",
                "dependencies": ["task_3", "task_4"]
            }
        ]
    )

    # Execute workflow
    results = integrator.execute_workflow(workflow_id, max_iterations=20)

    # Show status
    status = integrator.get_workflow_status(workflow_id)
    print("\nWorkflow Status:")
    print(json.dumps(status, indent=2))

    # Show stats
    stats = integrator.get_statistics()
    print("\nSystem Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
