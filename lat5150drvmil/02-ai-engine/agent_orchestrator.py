#!/usr/bin/env python3
"""
12-Factor Agent Orchestrator for Dynamic Agent Creation and Coordination

Implements the 12-factor-agents principles from humanlayer/12-factor-agents:
1. Natural Language to Tool Calls - LLMs generate structured outputs
2. Own Your Prompts - Direct control over prompts
3. Own Your Context Window - Active context management
4. Tools Are Structured Outputs - JSON schemas for tools
5. Unify Execution and Business State - Synchronized state
6. Launch/Pause/Resume - Simple agent lifecycle
7. Contact Humans with Tool Calls - Human-in-loop via tools
8. Own Your Control Flow - Explicit logic
9. Compact Errors into Context - Efficient error handling
10. Small, Focused Agents - Specialist agents
11. Trigger from Anywhere - Flexible invocation
12. Stateless Reducer Pattern - Pure state transformations

Features:
- Dynamic agent creation with specialized capabilities
- Inter-agent communication protocol
- Project-based orchestration  
- Unified state management
- Human-in-loop decision points
- Explicit control flow with pause/resume
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import json
import time
import uuid
from datetime import datetime


# ===== FACTOR 4: Tools Are Structured Outputs =====

@dataclass
class ToolSchema:
    """JSON schema definition for agent tools"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }


@dataclass
class ToolCall:
    """Structured output from LLM representing a tool invocation"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    """Result from tool execution"""
    call_id: str
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ===== FACTOR 5: Unify Execution and Business State =====

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """
    Unified state combining execution and business state.

    Factor 12: Stateless Reducer Pattern - this state is transformed
    by pure functions rather than mutated in place.
    """
    agent_id: str
    agent_type: str  # "code_specialist", "security_analyst", etc.
    status: AgentStatus

    # Execution state
    current_step: int = 0
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_history: List[ToolCall] = field(default_factory=list)
    tool_results_history: List[ToolResult] = field(default_factory=list)

    # Business state (domain-specific data)
    project_id: Optional[str] = None
    task_description: str = ""
    task_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Factor 3: Context window management
    max_context_tokens: int = 128000
    current_context_tokens: int = 0

    # Factor 7: Human contact points
    human_decision_points: List[Dict[str, Any]] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["status"] = self.status.value
        data["tool_calls_history"] = [asdict(tc) for tc in self.tool_calls_history]
        data["tool_results_history"] = [asdict(tr) for tr in self.tool_results_history]
        return data


# ===== FACTOR 1: Natural Language to Tool Calls =====

@dataclass
class AgentPrompt:
    """
    Factor 2: Own Your Prompts - direct control over prompt construction
    """
    system_prompt: str
    user_prompt: str
    available_tools: List[ToolSchema] = field(default_factory=list)
    context_messages: List[Dict[str, Any]] = field(default_factory=list)

    def build_messages(self) -> List[Dict[str, str]]:
        """Build message list for LLM API"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.context_messages)
        messages.append({"role": "user", "content": self.user_prompt})
        return messages

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get tool schemas for function calling"""
        return [tool.to_json_schema() for tool in self.available_tools]


# ===== FACTOR 10: Small, Focused Agents =====

class AgentSpecialization(Enum):
    """Predefined agent specializations"""
    CODE_SPECIALIST = "code_specialist"
    SECURITY_ANALYST = "security_analyst"
    DATA_ENGINEER = "data_engineer"
    THREAT_HUNTER = "threat_hunter"
    BLOCKCHAIN_INVESTIGATOR = "blockchain_investigator"
    OSINT_RESEARCHER = "osint_researcher"
    MALWARE_ANALYST = "malware_analyst"
    FORENSICS_EXPERT = "forensics_expert"
    ORCHESTRATOR = "orchestrator"
    CUSTOM = "custom"


# ===== INTER-AGENT COMMUNICATION =====

@dataclass
class AgentMessage:
    """Message for inter-agent communication"""
    from_agent: str
    to_agent: str
    message_type: str  # "task_request", "result", "status_update", "question"
    content: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    response_to: Optional[str] = None


class MessageBus:
    """
    Central message bus for inter-agent communication.
    Enables agents to coordinate without tight coupling.
    """

    def __init__(self):
        self.queues: Dict[str, List[AgentMessage]] = {}
        self.message_history: List[AgentMessage] = []

    def send(self, message: AgentMessage):
        """Send message to target agent queue"""
        if message.to_agent not in self.queues:
            self.queues[message.to_agent] = []

        self.queues[message.to_agent].append(message)
        self.message_history.append(message)

    def receive(self, agent_id: str, limit: int = 10) -> List[AgentMessage]:
        """Receive messages for an agent"""
        if agent_id not in self.queues:
            return []

        messages = self.queues[agent_id][:limit]
        self.queues[agent_id] = self.queues[agent_id][limit:]
        return messages

    def get_conversation(self, agent1: str, agent2: str) -> List[AgentMessage]:
        """Get message history between two agents"""
        return [
            msg for msg in self.message_history
            if (msg.from_agent == agent1 and msg.to_agent == agent2) or
               (msg.from_agent == agent2 and msg.to_agent == agent1)
        ]


# ===== FACTOR 10: Agent Factory for Specialist Creation =====

class AgentFactory:
    """
    Factory for creating specialized agents dynamically.

    Implements Factor 10: Small, Focused Agents
    Each agent is designed for a specific domain or task type.
    """

    @classmethod
    def create_agent(
        cls,
        specialization: Union[AgentSpecialization, str],
        task_description: str,
        project_id: Optional[str] = None
    ) -> AgentState:
        """Create a new specialized agent"""
        if isinstance(specialization, str):
            try:
                specialization = AgentSpecialization(specialization)
            except ValueError:
                specialization = AgentSpecialization.CUSTOM

        agent_id = f"{specialization.value}_{uuid.uuid4().hex[:8]}"

        state = AgentState(
            agent_id=agent_id,
            agent_type=specialization.value,
            status=AgentStatus.IDLE,
            task_description=task_description,
            project_id=project_id,
            max_context_tokens=128000,
            metadata={
                "specialization": specialization.value
            }
        )

        return state


# ===== FACTOR 6: Launch/Pause/Resume with Simple APIs =====

class AgentExecutor:
    """
    Manages agent execution lifecycle with pause/resume capabilities.

    Factor 6: Launch/Pause/Resume with Simple APIs
    Factor 8: Own Your Control Flow - explicit state transitions
    """

    def __init__(self):
        self.active_agents: Dict[str, AgentState] = {}
        self.tool_implementations: Dict[str, Callable] = {}

    def register_tool(self, tool_name: str, implementation: Callable):
        """Register a tool implementation"""
        self.tool_implementations[tool_name] = implementation

    def launch(self, state: AgentState) -> AgentState:
        """Launch an agent (Factor 12: Stateless reducer)"""
        new_state = AgentState(**state.to_dict())
        new_state.status = AgentStatus.RUNNING
        new_state.updated_at = time.time()
        self.active_agents[new_state.agent_id] = new_state
        return new_state

    def pause(self, agent_id: str) -> Optional[AgentState]:
        """Pause an agent's execution"""
        if agent_id not in self.active_agents:
            return None
        state = self.active_agents[agent_id]
        new_state = AgentState(**state.to_dict())
        new_state.status = AgentStatus.PAUSED
        new_state.updated_at = time.time()
        self.active_agents[agent_id] = new_state
        return new_state

    def resume(self, agent_id: str) -> Optional[AgentState]:
        """Resume a paused agent"""
        if agent_id not in self.active_agents:
            return None
        state = self.active_agents[agent_id]
        if state.status != AgentStatus.PAUSED:
            return None
        new_state = AgentState(**state.to_dict())
        new_state.status = AgentStatus.RUNNING
        new_state.updated_at = time.time()
        self.active_agents[agent_id] = new_state
        return new_state

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current agent state"""
        return self.active_agents.get(agent_id)

    def list_active_agents(self) -> List[str]:
        """List all active agent IDs"""
        return list(self.active_agents.keys())


# ===== PROJECT-BASED ORCHESTRATION =====

@dataclass
class Project:
    """Container for multi-agent projects"""
    project_id: str
    name: str
    description: str
    agent_ids: List[str] = field(default_factory=list)
    status: str = "active"
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ProjectOrchestrator:
    """
    Orchestrates multiple agents working on projects.

    Combines:
    - Factor 10: Small, focused agents
    - Inter-agent communication via MessageBus
    - Unified state management
    """

    def __init__(self, executor: AgentExecutor):
        self.executor = executor
        self.message_bus = MessageBus()
        self.projects: Dict[str, Project] = {}

    def create_project(
        self,
        name: str,
        description: str,
        required_specialists: List[Union[AgentSpecialization, str]]
    ) -> Project:
        """Create a new multi-agent project"""
        project_id = f"proj_{uuid.uuid4().hex[:12]}"

        agent_ids = []
        for spec in required_specialists:
            agent_state = AgentFactory.create_agent(
                specialization=spec,
                task_description=f"Contribute to project: {description}",
                project_id=project_id
            )
            launched_state = self.executor.launch(agent_state)
            agent_ids.append(launched_state.agent_id)

        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            agent_ids=agent_ids
        )

        self.projects[project_id] = project
        return project

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: Dict[str, Any]
    ):
        """Send message between agents"""
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        self.message_bus.send(message)

    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive project status"""
        if project_id not in self.projects:
            return None

        project = self.projects[project_id]
        agent_statuses = {}
        for agent_id in project.agent_ids:
            state = self.executor.get_state(agent_id)
            if state:
                agent_statuses[agent_id] = {
                    "type": state.agent_type,
                    "status": state.status.value,
                    "current_step": state.current_step
                }

        return {
            "project_id": project.project_id,
            "name": project.name,
            "status": project.status,
            "agents": agent_statuses,
            "results": project.results
        }


# ===== EXAMPLE USAGE =====

def example_usage():
    """Demonstrate 12-factor agent orchestration"""

    # Initialize executor
    executor = AgentExecutor()

    # Create orchestrator
    orchestrator = ProjectOrchestrator(executor)

    # Create a multi-agent project
    project = orchestrator.create_project(
        name="Security Audit of Web Application",
        description="Comprehensive security audit",
        required_specialists=[
            AgentSpecialization.CODE_SPECIALIST,
            AgentSpecialization.SECURITY_ANALYST,
            AgentSpecialization.OSINT_RESEARCHER
        ]
    )

    print(f"Created project: {project.project_id}")
    print(f"Agents: {project.agent_ids}")

    # Get project status
    status = orchestrator.get_project_status(project.project_id)
    print(f"Project status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    example_usage()

