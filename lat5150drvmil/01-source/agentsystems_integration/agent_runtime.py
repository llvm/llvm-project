#!/usr/bin/env python3
"""
LAT5150 DRVMIL - AgentSystems Containerized Agent Runtime
Secure agent execution with container isolation and audit logging

Based on: https://github.com/agentsystems/agentsystems
Approach: Container-based isolation with runtime credential injection
"""

import os
import sys
import json
import uuid
import hashlib
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] AgentRuntime: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Hash-chained audit log event"""
    timestamp: str
    action: str
    data: Dict[str, Any]
    previous_hash: str
    event_hash: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    image: str
    egress_allowlist: List[str]
    resource_limits: Dict[str, Any]
    security_profile: str
    version: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentExecution:
    """Agent execution result"""
    agent_name: str
    thread_id: str
    status: str
    start_time: str
    end_time: Optional[str]
    output: Any
    artifacts: List[str]
    logs: List[str]
    error: Optional[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class HashChainedAuditLogger:
    """
    Tamper-evident audit logging with hash chaining

    Each event contains hash of previous event, creating verifiable chain
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.chain: List[AuditEvent] = []
        self.previous_hash = "0" * 64  # Genesis hash

        # Load existing chain
        if self.log_path.exists():
            self._load_chain()

    def log_event(
        self,
        action: str,
        data: Dict[str, Any]
    ) -> AuditEvent:
        """Log an event and add to hash chain"""

        # Create event
        event_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "data": json.dumps(data, sort_keys=True),
            "previous_hash": self.previous_hash
        }

        # Compute event hash
        event_string = json.dumps(event_data, sort_keys=True)
        event_hash = hashlib.sha3_512(event_string.encode()).hexdigest()

        # Create audit event
        audit_event = AuditEvent(
            timestamp=event_data["timestamp"],
            action=action,
            data=data,
            previous_hash=self.previous_hash,
            event_hash=event_hash
        )

        # Add to chain
        self.chain.append(audit_event)
        self.previous_hash = event_hash

        # Persist to disk
        self._append_to_log(audit_event)

        return audit_event

    def verify_chain(self) -> bool:
        """Verify integrity of audit chain"""
        previous_hash = "0" * 64  # Genesis

        for event in self.chain:
            # Check previous_hash linkage
            if event.previous_hash != previous_hash:
                logger.error(f"Chain broken at event {event.timestamp}: previous_hash mismatch")
                return False

            # Recompute event hash
            event_data = {
                "timestamp": event.timestamp,
                "action": event.action,
                "data": json.dumps(event.data, sort_keys=True),
                "previous_hash": event.previous_hash
            }
            event_string = json.dumps(event_data, sort_keys=True)
            computed_hash = hashlib.sha3_512(event_string.encode()).hexdigest()

            # Verify hash
            if computed_hash != event.event_hash:
                logger.error(f"Chain broken at event {event.timestamp}: hash mismatch")
                return False

            previous_hash = event.event_hash

        return True

    def get_events(
        self,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit events"""
        events = self.chain[-limit:] if not action else [e for e in self.chain if e.action == action]
        return events[-limit:]

    def _append_to_log(self, event: AuditEvent):
        """Append event to log file"""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    def _load_chain(self):
        """Load existing chain from log file"""
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    event_data = json.loads(line.strip())
                    event = AuditEvent(**event_data)
                    self.chain.append(event)

            if self.chain:
                self.previous_hash = self.chain[-1].event_hash
                logger.info(f"Loaded {len(self.chain)} events from audit log")

                # Verify chain integrity
                if not self.verify_chain():
                    logger.error("Audit chain verification FAILED - possible tampering detected!")
                else:
                    logger.info("✅ Audit chain verified intact")

        except Exception as e:
            logger.error(f"Error loading audit chain: {e}")


class ContainerRuntime:
    """
    Container runtime for agent isolation

    Supports Docker and Podman
    """

    def __init__(self, runtime: str = "docker"):
        self.runtime = runtime  # docker or podman
        self._check_runtime()

    def _check_runtime(self):
        """Check if container runtime is available"""
        try:
            result = subprocess.run(
                [self.runtime, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                logger.info(f"Container runtime available: {result.stdout.strip()}")
            else:
                logger.error(f"Container runtime '{self.runtime}' not available")

        except Exception as e:
            logger.error(f"Error checking container runtime: {e}")

    async def run_agent(
        self,
        image: str,
        thread_id: str,
        env_vars: Dict[str, str],
        artifact_path: str,
        egress_allowlist: List[str],
        resource_limits: Dict[str, Any]
    ) -> subprocess.CompletedProcess:
        """
        Run agent in isolated container

        Args:
            image: Container image name
            thread_id: Unique thread identifier
            env_vars: Environment variables (includes injected credentials)
            artifact_path: Path to thread-scoped artifact storage
            egress_allowlist: Allowed egress domains
            resource_limits: CPU/memory limits

        Returns:
            CompletedProcess with stdout/stderr
        """

        # Build container run command
        cmd = [
            self.runtime, "run",
            "--rm",  # Remove container after exit
            "--read-only",  # Read-only root filesystem
            "--security-opt", "no-new-privileges",  # No privilege escalation
            "--cap-drop", "ALL",  # Drop all capabilities
        ]

        # Add resource limits
        if "cpu" in resource_limits:
            cmd.extend(["--cpus", str(resource_limits["cpu"])])
        if "memory" in resource_limits:
            cmd.extend(["--memory", resource_limits["memory"]])

        # Add environment variables (credentials injected here)
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Mount artifact storage (thread-scoped)
        cmd.extend(["-v", f"{artifact_path}:/artifacts"])

        # Network configuration
        # TODO: Add egress proxy for allowlist enforcement
        cmd.extend(["--network", "bridge"])

        # Container image
        cmd.append(image)

        # Execute container
        try:
            logger.info(f"Starting agent container: {image}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else ""
            )

            logger.info(f"Agent container completed with exit code {result.returncode}")
            return result

        except Exception as e:
            logger.error(f"Error running agent container: {e}")
            raise


class AgentOrchestrator:
    """
    Main agent orchestration system

    Integrates:
    - Container runtime
    - Credential injection
    - Audit logging
    - Thread-scoped storage
    """

    def __init__(
        self,
        artifact_base_path: str = "/opt/lat5150/artifacts",
        audit_log_path: str = "/opt/lat5150/audit/agent_audit.log",
        container_runtime: str = "docker"
    ):
        self.artifact_base_path = Path(artifact_base_path)
        self.artifact_base_path.mkdir(parents=True, exist_ok=True)

        self.audit_logger = HashChainedAuditLogger(audit_log_path)
        self.container_runtime = ContainerRuntime(container_runtime)

        self.agents: Dict[str, AgentConfig] = {}
        self.executions: Dict[str, AgentExecution] = {}

        # User credentials (injected at runtime, never stored in agent code)
        self.credentials: Dict[str, str] = {}

    def register_agent(self, agent_config: AgentConfig):
        """Register an agent for execution"""
        self.agents[agent_config.name] = agent_config
        logger.info(f"Registered agent: {agent_config.name}")

        # Log registration
        self.audit_logger.log_event(
            action="agent_registered",
            data={
                "agent_name": agent_config.name,
                "image": agent_config.image,
                "version": agent_config.version
            }
        )

    def configure_credentials(self, credentials: Dict[str, str]):
        """Configure user credentials for runtime injection"""
        self.credentials = credentials
        logger.info(f"Configured credentials for {len(credentials)} providers")

        # Log credential configuration (not the actual credentials!)
        self.audit_logger.log_event(
            action="credentials_configured",
            data={
                "providers": list(credentials.keys()),
                "count": len(credentials)
            }
        )

    async def invoke_agent(
        self,
        agent_name: str,
        task: Dict[str, Any],
        model_provider: Optional[str] = None
    ) -> AgentExecution:
        """
        Invoke an agent with container isolation

        Args:
            agent_name: Name of registered agent
            task: Task parameters
            model_provider: Model provider to use (anthropic, openai, ollama, etc.)

        Returns:
            AgentExecution result
        """

        # Validate agent exists
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")

        agent_config = self.agents[agent_name]

        # Generate thread ID
        thread_id = f"thread-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        # Create thread-scoped artifact directory
        thread_artifact_path = self.artifact_base_path / thread_id
        thread_artifact_path.mkdir(parents=True, exist_ok=True)

        # Log agent invocation
        self.audit_logger.log_event(
            action="agent_invoked",
            data={
                "agent_name": agent_name,
                "thread_id": thread_id,
                "task": task,
                "model_provider": model_provider
            }
        )

        # Prepare environment variables with injected credentials
        env_vars = {
            "THREAD_ID": thread_id,
            "ARTIFACTS_PATH": "/artifacts",
            "TASK": json.dumps(task)
        }

        # Inject model provider credentials
        if model_provider:
            if model_provider == "anthropic" and "anthropic_api_key" in self.credentials:
                env_vars["ANTHROPIC_API_KEY"] = self.credentials["anthropic_api_key"]
            elif model_provider == "openai" and "openai_api_key" in self.credentials:
                env_vars["OPENAI_API_KEY"] = self.credentials["openai_api_key"]
            elif model_provider == "ollama":
                env_vars["OLLAMA_ENDPOINT"] = self.credentials.get("ollama_endpoint", "http://localhost:11434")

        # Log credential injection (not the actual credentials!)
        self.audit_logger.log_event(
            action="credentials_injected",
            data={
                "thread_id": thread_id,
                "provider": model_provider,
                "credentials_provided": bool(model_provider and model_provider in ["anthropic", "openai", "ollama"])
            }
        )

        # Create execution record
        execution = AgentExecution(
            agent_name=agent_name,
            thread_id=thread_id,
            status="running",
            start_time=datetime.utcnow().isoformat() + "Z",
            end_time=None,
            output=None,
            artifacts=[],
            logs=[],
            error=None
        )

        self.executions[thread_id] = execution

        try:
            # Run agent in container
            start_time = time.time()

            result = await self.container_runtime.run_agent(
                image=agent_config.image,
                thread_id=thread_id,
                env_vars=env_vars,
                artifact_path=str(thread_artifact_path),
                egress_allowlist=agent_config.egress_allowlist,
                resource_limits=agent_config.resource_limits
            )

            end_time = time.time()
            duration = end_time - start_time

            # Update execution
            execution.status = "completed" if result.returncode == 0 else "failed"
            execution.end_time = datetime.utcnow().isoformat() + "Z"
            execution.output = result.stdout
            execution.error = result.stderr if result.returncode != 0 else None

            # Collect artifacts
            execution.artifacts = [
                str(f) for f in thread_artifact_path.iterdir()
            ]

            # Log completion
            self.audit_logger.log_event(
                action="agent_completed",
                data={
                    "thread_id": thread_id,
                    "status": execution.status,
                    "duration_seconds": duration,
                    "exit_code": result.returncode,
                    "artifacts_count": len(execution.artifacts)
                }
            )

            logger.info(f"Agent '{agent_name}' completed in {duration:.2f}s with status: {execution.status}")

            return execution

        except Exception as e:
            logger.error(f"Error invoking agent '{agent_name}': {e}")

            # Update execution with error
            execution.status = "error"
            execution.end_time = datetime.utcnow().isoformat() + "Z"
            execution.error = str(e)

            # Log error
            self.audit_logger.log_event(
                action="agent_error",
                data={
                    "thread_id": thread_id,
                    "error": str(e)
                }
            )

            return execution

    def get_execution(self, thread_id: str) -> Optional[AgentExecution]:
        """Get execution result by thread ID"""
        return self.executions.get(thread_id)

    def list_agents(self) -> List[AgentConfig]:
        """List registered agents"""
        return list(self.agents.values())

    def verify_audit_chain(self) -> bool:
        """Verify integrity of audit log chain"""
        is_valid = self.audit_logger.verify_chain()

        # Log verification result
        self.audit_logger.log_event(
            action="audit_chain_verified",
            data={
                "valid": is_valid,
                "event_count": len(self.audit_logger.chain)
            }
        )

        return is_valid

    def get_audit_events(
        self,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit events"""
        return self.audit_logger.get_events(action=action, limit=limit)


# Example usage
async def main():
    """Test agent orchestrator"""

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        artifact_base_path="/tmp/lat5150/artifacts",
        audit_log_path="/tmp/lat5150/audit/agent_audit.log"
    )

    # Configure credentials
    orchestrator.configure_credentials({
        "anthropic_api_key": "sk-ant-xxxxx",  # Example
        "openai_api_key": "sk-proj-xxxxx",    # Example
        "ollama_endpoint": "http://localhost:11434"
    })

    # Register example agent
    orchestrator.register_agent(AgentConfig(
        name="code-analyzer",
        image="code-analyzer:latest",
        egress_allowlist=[
            "api.anthropic.com",
            "api.openai.com",
            "github.com"
        ],
        resource_limits={
            "cpu": 2.0,
            "memory": "4g"
        },
        security_profile="default",
        version="1.0.0"
    ))

    # Example invocation (would need actual container image)
    # execution = await orchestrator.invoke_agent(
    #     agent_name="code-analyzer",
    #     task={"code_path": "/path/to/code", "analysis_type": "security"},
    #     model_provider="anthropic"
    # )
    # print(f"Execution result: {execution.status}")

    # Verify audit chain
    is_valid = orchestrator.verify_audit_chain()
    print(f"Audit chain valid: {is_valid}")

    # Get recent audit events
    events = orchestrator.get_audit_events(limit=10)
    print(f"\nRecent audit events ({len(events)}):")
    for event in events:
        print(f"  - {event.timestamp}: {event.action}")


if __name__ == "__main__":
    asyncio.run(main())
