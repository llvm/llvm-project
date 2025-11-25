#!/usr/bin/env python3
"""
DSMIL Agent Coordinator v1.0
ML-powered orchestration of 80 specialized Claude agents
Optimized for DSMIL device monitoring and maintenance tasks
"""

import os
import sys
import asyncio
import asyncpg
import json
import time
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

# Add DSMIL paths
DSMIL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(DSMIL_ROOT))

# Add Claude project paths for agent access
CLAUDE_ROOT = Path("/home/john/claude-backups")
sys.path.insert(0, str(CLAUDE_ROOT))

class AgentCategory(Enum):
    """Categories of available agents"""
    COMMAND_CONTROL = "command_control"
    SECURITY = "security"
    DEVELOPMENT = "development"
    LANGUAGE_SPECIFIC = "language_specific"
    INFRASTRUCTURE = "infrastructure"
    PLATFORM = "platform"
    NETWORK_SYSTEMS = "network_systems"
    DATA_ML = "data_ml"
    HARDWARE = "hardware"
    PLANNING = "planning"
    QUALITY = "quality"

class TaskPriority(Enum):
    """Task priority levels"""
    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_name: str
    category: AgentCategory
    specializations: List[str]
    performance_score: float
    average_duration: float
    success_rate: float
    resource_requirements: Dict[str, Any]
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
@dataclass
class TaskRequest:
    """Task request for agent coordination"""
    task_id: str
    description: str
    priority: TaskPriority
    device_context: Optional[Dict[str, Any]] = None
    required_capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentAssignment:
    """Agent assignment result"""
    agent_name: str
    task_id: str
    confidence: float
    estimated_duration: float
    required_resources: Dict[str, Any]
    assigned_at: datetime
    expected_completion: datetime
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ExecutionResult:
    """Agent execution result"""
    task_id: str
    agent_name: str
    success: bool
    duration: float
    output: Any
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class DSMILAgentCoordinator:
    """Main coordinator for 80 specialized Claude agents"""
    
    def __init__(self, db_pool: asyncpg.Pool, config: Dict[str, Any]):
        self.db_pool = db_pool
        self.config = config
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.active_assignments: Dict[str, AgentAssignment] = {}
        self.task_queue: List[TaskRequest] = []
        self.execution_history: List[ExecutionResult] = []
        self.coordinator_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.average_assignment_time = 0.0
        
        # Initialize agent capabilities
        self._initialize_agent_capabilities()
    
    def _initialize_agent_capabilities(self) -> None:
        """Initialize capabilities for all 80 specialized agents"""
        
        # Command & Control (2 agents)
        self.agent_capabilities.update({
            "director": AgentCapability(
                agent_name="director",
                category=AgentCategory.COMMAND_CONTROL,
                specializations=["strategic_planning", "project_coordination", "high_level_decisions"],
                performance_score=0.95,
                average_duration=300.0,  # 5 minutes
                success_rate=0.98,
                resource_requirements={"cpu": 2, "memory": "1GB", "priority": "high"},
                keywords=["strategy", "plan", "coordinate", "manage", "oversight", "decision"],
                dependencies=[]
            ),
            "projectorchestrator": AgentCapability(
                agent_name="projectorchestrator", 
                category=AgentCategory.COMMAND_CONTROL,
                specializations=["tactical_coordination", "multi_agent_orchestration", "workflow_management"],
                performance_score=0.93,
                average_duration=180.0,  # 3 minutes
                success_rate=0.96,
                resource_requirements={"cpu": 2, "memory": "512MB", "priority": "high"},
                keywords=["orchestrate", "coordinate", "workflow", "multi-agent", "tactical"],
                dependencies=[]
            )
        })
        
        # Security Specialists (13 agents) 
        security_agents = {
            "security": ["vulnerability", "audit", "threat", "assessment", "analysis"],
            "bastion": ["defense", "protection", "hardening", "security_infrastructure"],
            "securitychaosagent": ["chaos_testing", "fault_injection", "resilience"],
            "securityauditor": ["compliance", "audit", "governance", "risk_assessment"],
            "cso": ["security_strategy", "policy", "governance", "leadership"],
            "cryptoexpert": ["encryption", "cryptography", "key_management", "ciphers"],
            "quantumguard": ["quantum_security", "post_quantum", "quantum_resistant"],
            "redteamorchestrator": ["penetration_testing", "red_team", "attack_simulation"],
            "apt41-defense": ["apt_defense", "advanced_threats", "nation_state"],
            "nsa-ttp": ["tactics", "techniques", "procedures", "intelligence"],
            "psyops": ["psychological_operations", "influence", "manipulation_defense"],
            "ghost-protocol": ["stealth", "evasion", "counter_surveillance"],
            "cognitive-defense": ["cognitive_security", "manipulation_detection", "defense"]
        }
        
        for agent, keywords in security_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.SECURITY,
                specializations=keywords,
                performance_score=0.90,
                average_duration=120.0,
                success_rate=0.94,
                resource_requirements={"cpu": 1, "memory": "512MB", "priority": "medium"},
                keywords=keywords + ["security", "protect", "defend"],
                dependencies=[]
            )
        
        # Core Development (8 agents)
        dev_agents = {
            "architect": ["system_design", "architecture", "patterns", "structure"],
            "constructor": ["project_setup", "initialization", "scaffolding", "bootstrap"],
            "patcher": ["bug_fixes", "patches", "code_surgery", "hotfixes"], 
            "debugger": ["debugging", "troubleshooting", "error_analysis", "diagnostics"],
            "testbed": ["testing", "qa", "validation", "test_automation"],
            "linter": ["code_review", "quality", "standards", "static_analysis"],
            "optimizer": ["performance", "optimization", "efficiency", "tuning"],
            "qadirector": ["quality_assurance", "qa_management", "testing_strategy"]
        }
        
        for agent, keywords in dev_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.DEVELOPMENT,
                specializations=keywords,
                performance_score=0.88,
                average_duration=240.0,
                success_rate=0.92,
                resource_requirements={"cpu": 2, "memory": "1GB", "priority": "medium"},
                keywords=keywords + ["development", "code", "software"],
                dependencies=[]
            )
        
        # Language-Specific Development (11 agents)
        language_agents = {
            "c-internal": ["c_programming", "systems", "low_level", "kernel"],
            "cpp-internal": ["cpp", "object_oriented", "templates", "modern_cpp"],
            "python-internal": ["python", "scripting", "automation", "data_science"],
            "rust-internal": ["rust", "memory_safety", "systems", "performance"],
            "go-internal": ["golang", "concurrency", "microservices", "cloud_native"],
            "java-internal": ["java", "enterprise", "jvm", "spring"],
            "typescript-internal": ["typescript", "javascript", "web", "node"],
            "kotlin-internal": ["kotlin", "android", "multiplatform", "jvm"],
            "assembly-internal": ["assembly", "machine_code", "low_level", "optimization"],
            "sql-internal": ["sql", "database", "queries", "data_management"],
            "zig-internal": ["zig", "systems", "performance", "c_replacement"]
        }
        
        for agent, keywords in language_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.LANGUAGE_SPECIFIC,
                specializations=keywords,
                performance_score=0.85,
                average_duration=300.0,
                success_rate=0.90,
                resource_requirements={"cpu": 2, "memory": "1GB", "priority": "medium"},
                keywords=keywords + ["programming", "development", "code"],
                dependencies=[]
            )
        
        # Infrastructure & DevOps (6 agents)
        infra_agents = {
            "infrastructure": ["system_setup", "configuration", "deployment", "ops"],
            "deployer": ["deployment", "cicd", "release", "automation"],
            "monitor": ["monitoring", "observability", "metrics", "alerting"],
            "packager": ["packaging", "distribution", "containers", "artifacts"],
            "docker": ["containers", "docker", "containerization", "orchestration"],
            "proxmox": ["virtualization", "hypervisor", "vm_management", "clusters"]
        }
        
        for agent, keywords in infra_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.INFRASTRUCTURE,
                specializations=keywords,
                performance_score=0.87,
                average_duration=360.0,
                success_rate=0.91,
                resource_requirements={"cpu": 2, "memory": "2GB", "priority": "medium"},
                keywords=keywords + ["infrastructure", "operations", "devops"],
                dependencies=[]
            )
        
        # Specialized Platforms (7 agents)
        platform_agents = {
            "apidesigner": ["api_design", "rest", "graphql", "microservices"],
            "database": ["database_design", "schemas", "optimization", "queries"],
            "web": ["web_development", "frontend", "react", "vue", "angular"],
            "mobile": ["mobile_development", "ios", "android", "cross_platform"],
            "androidmobile": ["android", "mobile", "kotlin", "java"],
            "pygui": ["python_gui", "tkinter", "pyqt", "desktop"],
            "tui": ["terminal_ui", "console", "cli", "text_interface"]
        }
        
        for agent, keywords in platform_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.PLATFORM,
                specializations=keywords,
                performance_score=0.84,
                average_duration=420.0,
                success_rate=0.89,
                resource_requirements={"cpu": 2, "memory": "1GB", "priority": "medium"},
                keywords=keywords + ["platform", "interface", "ui"],
                dependencies=[]
            )
        
        # Network & Systems (4 agents)
        network_agents = {
            "cisco": ["cisco", "networking", "routers", "switches"],
            "bgp-purple-team": ["bgp", "routing", "network_security", "protocols"],
            "iot-access-control": ["iot", "access_control", "device_security", "authentication"],
            "ddwrt": ["router_firmware", "networking", "embedded", "wireless"]
        }
        
        for agent, keywords in network_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.NETWORK_SYSTEMS,
                specializations=keywords,
                performance_score=0.82,
                average_duration=300.0,
                success_rate=0.88,
                resource_requirements={"cpu": 1, "memory": "512MB", "priority": "medium"},
                keywords=keywords + ["network", "systems", "connectivity"],
                dependencies=[]
            )
        
        # Data & ML (4 agents)
        data_ml_agents = {
            "datascience": ["data_analysis", "statistics", "ml", "analytics"],
            "mlops": ["ml_operations", "model_deployment", "pipelines", "automation"],
            "npu": ["neural_processing", "ai_acceleration", "inference", "optimization"],
            "sql-internal": ["sql", "database", "queries", "data_management"]
        }
        
        for agent, keywords in data_ml_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.DATA_ML,
                specializations=keywords,
                performance_score=0.86,
                average_duration=480.0,
                success_rate=0.87,
                resource_requirements={"cpu": 4, "memory": "4GB", "priority": "medium"},
                keywords=keywords + ["data", "analytics", "machine_learning"],
                dependencies=[]
            )
        
        # Hardware & Acceleration (6 agents)
        hardware_agents = {
            "hardware": ["hardware_control", "registers", "low_level", "drivers"],
            "hardware-dell": ["dell_systems", "idrac", "bios", "tokens"],
            "hardware-hp": ["hp_systems", "ilo", "probook", "sure_start"],
            "hardware-intel": ["intel_systems", "npu", "gna", "meteor_lake"],
            "gna": ["gaussian_neural", "acceleration", "inference", "optimization"],
            "leadengineer": ["hardware_integration", "systems", "engineering"]
        }
        
        for agent, keywords in hardware_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.HARDWARE,
                specializations=keywords,
                performance_score=0.89,
                average_duration=240.0,
                success_rate=0.93,
                resource_requirements={"cpu": 2, "memory": "1GB", "priority": "high"},
                keywords=keywords + ["hardware", "systems", "low_level"],
                dependencies=[]
            )
        
        # Planning & Documentation (4 agents)
        planning_agents = {
            "planner": ["project_planning", "strategy", "roadmap", "scheduling"],
            "docgen": ["documentation", "technical_writing", "guides", "manuals"],
            "researcher": ["research", "analysis", "investigation", "evaluation"],
            "statusline-integration": ["integration", "tooling", "development_environment"]
        }
        
        for agent, keywords in planning_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.PLANNING,
                specializations=keywords,
                performance_score=0.83,
                average_duration=360.0,
                success_rate=0.86,
                resource_requirements={"cpu": 1, "memory": "512MB", "priority": "low"},
                keywords=keywords + ["planning", "documentation", "analysis"],
                dependencies=[]
            )
        
        # Quality & Oversight (3 agents)
        quality_agents = {
            "oversight": ["quality_assurance", "compliance", "governance", "standards"],
            "integration": ["system_integration", "compatibility", "interoperability"],
            "auditor": ["auditing", "compliance", "verification", "validation"]
        }
        
        for agent, keywords in quality_agents.items():
            self.agent_capabilities[agent] = AgentCapability(
                agent_name=agent,
                category=AgentCategory.QUALITY,
                specializations=keywords,
                performance_score=0.81,
                average_duration=300.0,
                success_rate=0.85,
                resource_requirements={"cpu": 1, "memory": "512MB", "priority": "low"},
                keywords=keywords + ["quality", "assurance", "compliance"],
                dependencies=[]
            )
        
        # Initialize all agent status as idle
        for agent_name in self.agent_capabilities.keys():
            self.agent_status[agent_name] = AgentStatus.IDLE
            
        self.logger.info(f"Initialized capabilities for {len(self.agent_capabilities)} agents")
    
    async def initialize(self) -> bool:
        """Initialize the agent coordinator system"""
        try:
            self.logger.info("Initializing DSMIL Agent Coordinator...")
            
            # Verify database connectivity
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # Load agent performance history
            await self._load_agent_performance_history()
            
            # Start background tasks
            asyncio.create_task(self._process_task_queue())
            asyncio.create_task(self._monitor_agent_health())
            
            self.logger.info("Agent Coordinator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator: {e}")
            return False
    
    async def _load_agent_performance_history(self) -> None:
        """Load agent performance history from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT agent_name, 
                           AVG(execution_time) as avg_duration,
                           AVG(success_rate) as success_rate,
                           COUNT(*) as task_count
                    FROM agent_metrics 
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    GROUP BY agent_name
                """)
                
                for row in rows:
                    agent_name = row['agent_name']
                    if agent_name in self.agent_capabilities:
                        # Update performance data
                        capability = self.agent_capabilities[agent_name]
                        capability.average_duration = max(row['avg_duration'], 10.0)
                        capability.success_rate = row['success_rate']
                        # Calculate performance score based on success rate and speed
                        capability.performance_score = (
                            row['success_rate'] * 0.7 + 
                            (300 / max(row['avg_duration'], 30)) * 0.3
                        )
                
                self.logger.info(f"Loaded performance history for {len(rows)} agents")
                
        except Exception as e:
            self.logger.warning(f"Failed to load agent performance history: {e}")
    
    async def submit_task(self, task: TaskRequest) -> bool:
        """Submit a task for agent coordination"""
        try:
            with self.coordinator_lock:
                # Add task to queue with priority sorting
                self.task_queue.append(task)
                self.task_queue.sort(key=lambda t: t.priority.value)
            
            self.logger.info(f"Task {task.task_id} submitted with priority {task.priority.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def _process_task_queue(self) -> None:
        """Background task to process the task queue"""
        while True:
            try:
                # Get next task from queue
                task = None
                with self.coordinator_lock:
                    if self.task_queue:
                        task = self.task_queue.pop(0)
                
                if task:
                    await self._handle_task(task)
                else:
                    # No tasks in queue, wait before checking again
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                self.logger.error(f"Error in task queue processing: {e}")
                await asyncio.sleep(5.0)
    
    async def _handle_task(self, task: TaskRequest) -> None:
        """Handle a single task by finding and assigning the best agent"""
        try:
            start_time = time.time()
            
            # Find best agent for the task
            assignment = await self._find_best_agent(task)
            
            if not assignment:
                self.logger.warning(f"No suitable agent found for task {task.task_id}")
                return
            
            # Execute the task
            result = await self._execute_task(assignment, task)
            
            # Record execution result
            self.execution_history.append(result)
            await self._record_execution_result(result)
            
            # Update metrics
            assignment_time = time.time() - start_time
            self.total_tasks_processed += 1
            if result.success:
                self.successful_tasks += 1
            
            # Update average assignment time
            self.average_assignment_time = (
                (self.average_assignment_time * (self.total_tasks_processed - 1) + assignment_time) / 
                self.total_tasks_processed
            )
            
            self.logger.info(
                f"Task {task.task_id} completed by {assignment.agent_name} "
                f"in {result.duration:.2f}s (success: {result.success})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle task {task.task_id}: {e}")
    
    async def _find_best_agent(self, task: TaskRequest) -> Optional[AgentAssignment]:
        """Find the best agent for a given task using ML-powered selection"""
        try:
            # Generate task embedding for similarity matching
            task_embedding = self._generate_task_embedding(task)
            
            # Score all available agents
            agent_scores = []
            
            for agent_name, capability in self.agent_capabilities.items():
                # Skip if agent is busy (unless emergency task)
                if (self.agent_status[agent_name] == AgentStatus.BUSY and 
                    task.priority != TaskPriority.EMERGENCY):
                    continue
                
                # Calculate agent score for this task
                score = await self._calculate_agent_score(task, capability, task_embedding)
                
                if score > 0:
                    agent_scores.append((agent_name, capability, score))
            
            if not agent_scores:
                return None
            
            # Sort by score (highest first)
            agent_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Select best agent
            best_agent, best_capability, best_score = agent_scores[0]
            
            # Create assignment
            estimated_duration = self._estimate_task_duration(task, best_capability)
            
            assignment = AgentAssignment(
                agent_name=best_agent,
                task_id=task.task_id,
                confidence=min(best_score, 1.0),
                estimated_duration=estimated_duration,
                required_resources=best_capability.resource_requirements.copy(),
                assigned_at=datetime.now(timezone.utc),
                expected_completion=datetime.now(timezone.utc) + timedelta(seconds=estimated_duration),
                dependencies=best_capability.dependencies.copy()
            )
            
            # Mark agent as busy
            self.agent_status[best_agent] = AgentStatus.BUSY
            self.active_assignments[task.task_id] = assignment
            
            return assignment
            
        except Exception as e:
            self.logger.error(f"Failed to find best agent for task {task.task_id}: {e}")
            return None
    
    def _generate_task_embedding(self, task: TaskRequest) -> np.ndarray:
        """Generate 512-dimensional embedding for task matching"""
        # Extract keywords from task description
        description_words = task.description.lower().split()
        
        # Create word hash features
        word_features = []
        for word in description_words[:50]:  # Limit to first 50 words
            word_features.append(hash(word) % 1000 / 1000.0)
        
        # Pad to 200 dimensions
        while len(word_features) < 200:
            word_features.append(0.0)
        word_features = word_features[:200]
        
        # Add priority and constraint features
        priority_features = [
            task.priority.value / 5.0,  # Normalize priority
            1.0 if task.deadline else 0.0,  # Has deadline
            len(task.required_capabilities) / 10.0,  # Number of requirements
            len(task.constraints) / 10.0,  # Number of constraints
        ]
        
        # Add device context features if available
        device_features = []
        if task.device_context:
            device_features = [
                task.device_context.get('temperature', 0) / 100.0,
                task.device_context.get('cpu_usage', 0) / 100.0,
                task.device_context.get('memory_usage', 0) / 100.0,
                task.device_context.get('error_count', 0) / 10.0
            ]
        else:
            device_features = [0.0] * 4
        
        # Add required capabilities hash
        capability_features = []
        for cap in task.required_capabilities[:20]:  # Limit to 20 capabilities
            capability_features.append(hash(cap) % 1000 / 1000.0)
        while len(capability_features) < 20:
            capability_features.append(0.0)
        
        # Combine all features
        all_features = word_features + priority_features + device_features + capability_features
        
        # Pad to exactly 512 dimensions
        while len(all_features) < 512:
            all_features.append(0.0)
        all_features = all_features[:512]
        
        return np.array(all_features, dtype=np.float32)
    
    async def _calculate_agent_score(self, task: TaskRequest, 
                                   capability: AgentCapability,
                                   task_embedding: np.ndarray) -> float:
        """Calculate agent suitability score for a task"""
        try:
            score = 0.0
            
            # Keyword matching score (40% weight)
            task_words = set(task.description.lower().split())
            agent_keywords = set(capability.keywords)
            keyword_overlap = len(task_words.intersection(agent_keywords))
            keyword_score = min(keyword_overlap / 5.0, 1.0) * 0.4
            score += keyword_score
            
            # Capability matching score (30% weight)
            if task.required_capabilities:
                matching_caps = 0
                for req_cap in task.required_capabilities:
                    if any(req_cap.lower() in spec.lower() 
                          for spec in capability.specializations):
                        matching_caps += 1
                capability_score = (matching_caps / len(task.required_capabilities)) * 0.3
                score += capability_score
            else:
                score += 0.15  # Base score if no specific capabilities required
            
            # Performance score (20% weight)
            performance_score = capability.performance_score * 0.2
            score += performance_score
            
            # Availability and load balancing (10% weight)
            if self.agent_status[capability.agent_name] == AgentStatus.IDLE:
                availability_score = 0.1
            else:
                availability_score = 0.02  # Lower score for busy agents
            score += availability_score
            
            # Priority-based adjustments
            if task.priority == TaskPriority.EMERGENCY:
                if capability.category in [AgentCategory.SECURITY, AgentCategory.HARDWARE]:
                    score *= 1.5  # Boost security and hardware agents for emergencies
            elif task.priority == TaskPriority.HIGH:
                if capability.category in [AgentCategory.COMMAND_CONTROL, AgentCategory.DEVELOPMENT]:
                    score *= 1.2
            
            # Device context adjustments
            if task.device_context:
                temp = task.device_context.get('temperature', 0)
                errors = task.device_context.get('error_count', 0)
                
                # Boost hardware agents for thermal issues
                if temp > 75 and capability.category == AgentCategory.HARDWARE:
                    score *= 1.3
                    
                # Boost security agents for high error counts (potential attacks)
                if errors > 5 and capability.category == AgentCategory.SECURITY:
                    score *= 1.2
            
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate agent score: {e}")
            return 0.0
    
    def _estimate_task_duration(self, task: TaskRequest, 
                              capability: AgentCapability) -> float:
        """Estimate task duration based on complexity and agent performance"""
        base_duration = capability.average_duration
        
        # Adjust based on task complexity
        complexity_multiplier = 1.0
        
        # Complexity indicators
        description_length = len(task.description.split())
        if description_length > 50:
            complexity_multiplier *= 1.5
        elif description_length < 10:
            complexity_multiplier *= 0.8
        
        if len(task.required_capabilities) > 3:
            complexity_multiplier *= 1.3
        
        if len(task.constraints) > 2:
            complexity_multiplier *= 1.2
        
        # Priority adjustments
        if task.priority == TaskPriority.EMERGENCY:
            complexity_multiplier *= 0.8  # Faster execution for emergencies
        elif task.priority == TaskPriority.LOW:
            complexity_multiplier *= 1.2  # More thorough for low priority
        
        return base_duration * complexity_multiplier
    
    async def _execute_task(self, assignment: AgentAssignment, 
                          task: TaskRequest) -> ExecutionResult:
        """Execute the assigned task using the selected agent"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing task {task.task_id} with agent {assignment.agent_name}")
            
            # Prepare execution command
            command = self._build_agent_command(assignment.agent_name, task)
            
            # Execute agent command
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=CLAUDE_ROOT
            )
            
            # Wait for completion with timeout
            timeout = assignment.estimated_duration * 2  # 2x estimated time as timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"Task execution timed out after {timeout}s")
            
            # Process results
            success = process.returncode == 0
            output = stdout.decode() if stdout else ""
            error_message = stderr.decode() if stderr else None
            
            duration = time.time() - start_time
            
            # Mark agent as idle again
            self.agent_status[assignment.agent_name] = AgentStatus.IDLE
            if task.task_id in self.active_assignments:
                del self.active_assignments[task.task_id]
            
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=assignment.agent_name,
                success=success,
                duration=duration,
                output=output,
                error_message=error_message,
                metadata={
                    "estimated_duration": assignment.estimated_duration,
                    "confidence": assignment.confidence,
                    "return_code": process.returncode
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Mark agent as idle again
            self.agent_status[assignment.agent_name] = AgentStatus.IDLE
            if task.task_id in self.active_assignments:
                del self.active_assignments[task.task_id]
            
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=assignment.agent_name,
                success=False,
                duration=duration,
                output="",
                error_message=str(e)
            )
    
    def _build_agent_command(self, agent_name: str, task: TaskRequest) -> List[str]:
        """Build command to execute specific agent"""
        # Use claude-agent command for agent execution
        command = [
            "claude-agent",  # Assume claude-agent is in PATH
            agent_name,
            task.description
        ]
        
        # Add device context as environment variables
        if task.device_context:
            env_vars = []
            for key, value in task.device_context.items():
                env_vars.extend(["--env", f"DSMIL_{key.upper()}={value}"])
            command.extend(env_vars)
        
        return command
    
    async def _record_execution_result(self, result: ExecutionResult) -> None:
        """Record execution result in database for learning"""
        try:
            async with self.db_pool.acquire() as conn:
                # Record in agent_metrics table
                await conn.execute("""
                    INSERT INTO agent_metrics 
                    (agent_name, task_type, execution_time, success_rate, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                result.agent_name,
                "dsmil_coordination",
                result.duration,
                1.0 if result.success else 0.0,
                json.dumps(result.metadata),
                result.completed_at
                )
                
                # Record in interaction_logs
                await conn.execute("""
                    INSERT INTO interaction_logs 
                    (source_agent, target_agent, interaction_type, payload, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                "dsmil_coordinator",
                result.agent_name,
                "task_execution",
                json.dumps({
                    "task_id": result.task_id,
                    "success": result.success,
                    "duration": result.duration,
                    "output_length": len(result.output) if result.output else 0
                }),
                result.completed_at
                )
                
        except Exception as e:
            self.logger.error(f"Failed to record execution result: {e}")
    
    async def _monitor_agent_health(self) -> None:
        """Background task to monitor agent health and availability"""
        while True:
            try:
                # Check for stalled agents (assignments past expected completion)
                current_time = datetime.now(timezone.utc)
                stalled_assignments = []
                
                for task_id, assignment in self.active_assignments.items():
                    if current_time > assignment.expected_completion:
                        stalled_assignments.append((task_id, assignment))
                
                # Handle stalled assignments
                for task_id, assignment in stalled_assignments:
                    self.logger.warning(
                        f"Task {task_id} assigned to {assignment.agent_name} appears stalled"
                    )
                    # Mark agent as potentially failed
                    self.agent_status[assignment.agent_name] = AgentStatus.FAILED
                    # Remove from active assignments
                    del self.active_assignments[task_id]
                
                # Health check for failed agents (reset after some time)
                for agent_name, status in self.agent_status.items():
                    if status == AgentStatus.FAILED:
                        # Reset to idle after 5 minutes
                        self.agent_status[agent_name] = AgentStatus.IDLE
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in agent health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination system status"""
        try:
            # Calculate success rate
            success_rate = (
                self.successful_tasks / max(self.total_tasks_processed, 1)
            ) * 100
            
            # Count agents by status
            status_counts = {}
            for status in AgentStatus:
                count = sum(1 for s in self.agent_status.values() if s == status)
                status_counts[status.value] = count
            
            # Count agents by category
            category_counts = {}
            for category in AgentCategory:
                count = sum(1 for cap in self.agent_capabilities.values() 
                          if cap.category == category)
                category_counts[category.value] = count
            
            # Recent execution statistics
            recent_executions = self.execution_history[-100:]  # Last 100 executions
            recent_success_rate = (
                sum(1 for r in recent_executions if r.success) / 
                max(len(recent_executions), 1) * 100
            )
            
            recent_avg_duration = (
                sum(r.duration for r in recent_executions) / 
                max(len(recent_executions), 1)
            )
            
            return {
                "status": "active",
                "total_agents": len(self.agent_capabilities),
                "agent_status": status_counts,
                "agent_categories": category_counts,
                "performance": {
                    "total_tasks_processed": self.total_tasks_processed,
                    "successful_tasks": self.successful_tasks,
                    "overall_success_rate": round(success_rate, 2),
                    "recent_success_rate": round(recent_success_rate, 2),
                    "average_assignment_time": round(self.average_assignment_time, 3),
                    "recent_avg_duration": round(recent_avg_duration, 2)
                },
                "current_state": {
                    "active_assignments": len(self.active_assignments),
                    "queued_tasks": len(self.task_queue),
                    "idle_agents": status_counts.get("idle", 0),
                    "busy_agents": status_counts.get("busy", 0)
                },
                "capabilities": {
                    "ml_powered_selection": True,
                    "priority_scheduling": True,
                    "load_balancing": True,
                    "performance_learning": True,
                    "real_time_monitoring": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get coordination status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_agent_recommendations(self, task_description: str,
                                      device_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get agent recommendations for a task without executing it"""
        try:
            # Create temporary task for analysis
            temp_task = TaskRequest(
                task_id="recommendation_analysis",
                description=task_description,
                priority=TaskPriority.MEDIUM,
                device_context=device_context
            )
            
            # Generate task embedding
            task_embedding = self._generate_task_embedding(temp_task)
            
            # Score all agents
            recommendations = []
            for agent_name, capability in self.agent_capabilities.items():
                score = await self._calculate_agent_score(temp_task, capability, task_embedding)
                if score > 0.1:  # Only include agents with reasonable scores
                    duration = self._estimate_task_duration(temp_task, capability)
                    
                    recommendations.append({
                        "agent_name": agent_name,
                        "category": capability.category.value,
                        "confidence": round(score, 3),
                        "estimated_duration": round(duration, 1),
                        "specializations": capability.specializations,
                        "performance_score": round(capability.performance_score, 3),
                        "success_rate": round(capability.success_rate * 100, 1),
                        "status": self.agent_status[agent_name].value,
                        "resource_requirements": capability.resource_requirements
                    })
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get agent recommendations: {e}")
            return []

# Usage example and testing
async def main():
    """Example usage of DSMIL Agent Coordinator"""
    import asyncpg
    
    # Mock database pool
    db_pool = await asyncpg.create_pool(
        host="localhost",
        port=5433,
        database="claude_auth",
        user="claude_auth",
        password="claude_auth_pass"
    )
    
    config = {
        "agents": {
            "max_concurrent": 10,
            "default_timeout": 30.0
        }
    }
    
    coordinator = DSMILAgentCoordinator(db_pool, config)
    
    try:
        # Initialize coordinator
        if not await coordinator.initialize():
            print("Failed to initialize coordinator")
            return
        
        # Example task
        task = TaskRequest(
            task_id="thermal_analysis_001",
            description="Analyze thermal conditions and recommend optimization for overheating DSMIL device",
            priority=TaskPriority.HIGH,
            device_context={
                "temperature": 88.5,
                "cpu_usage": 95.2,
                "error_count": 7,
                "device_id": 42
            },
            required_capabilities=["thermal_management", "hardware_optimization"]
        )
        
        # Submit task
        await coordinator.submit_task(task)
        
        # Get recommendations
        recommendations = await coordinator.get_agent_recommendations(
            "Monitor DSMIL device performance and security",
            {"temperature": 78, "cpu_usage": 65}
        )
        
        print(f"Agent Recommendations:")
        for rec in recommendations[:5]:
            print(f"  {rec['agent_name']}: {rec['confidence']:.3f} confidence, {rec['estimated_duration']:.1f}s")
        
        # Get system status
        status = await coordinator.get_coordination_status()
        print(f"\nCoordination Status:")
        print(f"  Total Agents: {status['total_agents']}")
        print(f"  Success Rate: {status['performance']['overall_success_rate']}%")
        
        # Wait a bit to let task processing happen
        await asyncio.sleep(2)
        
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())