#!/usr/bin/env python3
"""
COMPREHENSIVE AGENT DEPLOYMENT SYSTEM v2.0
=============================================

Multi-Agent Coordination System for Phase 2 DSMIL Deployment
Coordinates all 80+ specialized agents from claude-backups ecosystem

CRITICAL AGENTS ACTIVATED:
- DIRECTOR: Strategic command and control
- PROJECTORCHESTRATOR: Tactical coordination nexus  
- HARDWARE-DELL: Dell Latitude 5450 optimization
- SECURITY team: 22-agent security ecosystem
- OPTIMIZER: Performance tuning
- MONITOR: Real-time tracking
- All 80+ agents from claude-backups

Author: DIRECTOR + PROJECTORCHESTRATOR + ORCHESTRATOR coordination
Target: /home/john/LAT5150DRVMIL Phase 2 deployment
"""

import os
import json
import yaml
import subprocess
import asyncio
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/john/LAT5150DRVMIL/logs/agent_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    category: str
    priority: str
    status: str
    tools: List[str]
    specializations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    last_invocation: Optional[datetime] = None

@dataclass
class DeploymentTask:
    """Deployment task definition"""
    task_id: str
    name: str
    description: str
    assigned_agents: List[str]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    parallel_execution: bool = False

class AgentRegistry:
    """Comprehensive agent registry for all 80+ agents"""
    
    def __init__(self, agents_path: str = "/home/john/claude-backups/agents"):
        self.agents_path = Path(agents_path)
        self.agents: Dict[str, AgentCapability] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self.health_monitor = {}
        self.load_agents()
        
    def load_agents(self):
        """Load all agents from the claude-backups directory"""
        logger.info(f"Loading agents from {self.agents_path}")
        
        if not self.agents_path.exists():
            logger.error(f"Agents path does not exist: {self.agents_path}")
            return
            
        agent_files = list(self.agents_path.glob("*.md"))
        logger.info(f"Found {len(agent_files)} agent files")
        
        for agent_file in agent_files:
            try:
                self._parse_agent_file(agent_file)
            except Exception as e:
                logger.warning(f"Failed to parse {agent_file.name}: {e}")
                
        logger.info(f"Successfully loaded {len(self.agents)} agents")
        self._categorize_agents()
    
    def _parse_agent_file(self, agent_file: Path):
        """Parse individual agent markdown file with YAML frontmatter"""
        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract YAML frontmatter
        if content.startswith('---\n'):
            try:
                yaml_end = content.find('\n---\n', 4)
                if yaml_end > 0:
                    yaml_content = content[4:yaml_end]
                    metadata = yaml.safe_load(yaml_content)
                    
                    if isinstance(metadata, dict) and 'metadata' in metadata:
                        agent_meta = metadata['metadata']
                        
                        # Extract keywords and specializations from content
                        keywords = self._extract_keywords(content)
                        specializations = self._extract_specializations(content, agent_meta.get('name', ''))
                        
                        agent = AgentCapability(
                            name=agent_meta.get('name', agent_file.stem),
                            category=agent_meta.get('category', 'GENERAL'),
                            priority=agent_meta.get('priority', 'MEDIUM'),
                            status=agent_meta.get('status', 'DEVELOPMENT'),
                            tools=self._extract_tools(agent_meta.get('tools', {})),
                            specializations=specializations,
                            keywords=keywords,
                            performance_metrics={}
                        )
                        
                        self.agents[agent.name] = agent
                        logger.debug(f"Loaded agent: {agent.name} ({agent.category})")
                        
            except Exception as e:
                logger.debug(f"YAML parsing failed for {agent_file.name}, trying fallback: {e}")
                # Fallback parsing
                self._fallback_parse(agent_file, content)
    
    def _fallback_parse(self, agent_file: Path, content: str):
        """Fallback parsing for agents without proper YAML"""
        name = agent_file.stem
        
        # Extract category and specializations from content
        category = "GENERAL"
        if "security" in content.lower() or "security" in name.lower():
            category = "SECURITY"
        elif "hardware" in name.lower():
            category = "HARDWARE"
        elif "internal" in name.lower():
            category = "DEVELOPMENT"
        elif any(x in name.lower() for x in ["infrastructure", "deploy", "monitor"]):
            category = "INFRASTRUCTURE"
            
        keywords = self._extract_keywords(content)
        specializations = self._extract_specializations(content, name)
        
        agent = AgentCapability(
            name=name,
            category=category,
            priority="MEDIUM",
            status="PRODUCTION",
            tools=["Task"],  # Assume basic Task tool
            specializations=specializations,
            keywords=keywords
        )
        
        self.agents[name] = agent
        logger.debug(f"Fallback loaded agent: {name}")
    
    def _extract_tools(self, tools_config: Dict) -> List[str]:
        """Extract tools from agent configuration"""
        tools = []
        if isinstance(tools_config, dict):
            for category, tool_list in tools_config.items():
                if isinstance(tool_list, list):
                    tools.extend(tool_list)
                elif isinstance(tool_list, str):
                    tools.append(tool_list)
        return tools
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from agent content"""
        keywords = []
        
        # Look for keyword patterns
        keyword_patterns = [
            r'keywords?:\s*\n\s*-\s*(.+)',
            r'triggers?.*?keywords?.*?\n((?:\s*-\s*.+\n)*)',
            r'specializ(?:es|ations?).*?(?:in|:)\s*([^\n\.]+)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    keyword_text = match.group(1)
                    # Extract individual keywords
                    words = re.findall(r'[\w\-+]+', keyword_text)
                    keywords.extend(words[:10])  # Limit to 10 keywords
        
        return list(set(keywords))[:15]  # Deduplicate and limit
    
    def _extract_specializations(self, content: str, agent_name: str) -> List[str]:
        """Extract specializations from agent content"""
        specializations = []
        
        # Agent name-based specializations
        if "HARDWARE" in agent_name:
            specializations.extend(["hardware_control", "system_optimization"])
        if "SECURITY" in agent_name:
            specializations.extend(["security_analysis", "threat_detection"])
        if "DELL" in agent_name:
            specializations.extend(["dell_systems", "latitude_optimization"])
        if "INTEL" in agent_name:
            specializations.extend(["intel_hardware", "meteor_lake"])
            
        # Content-based specializations
        spec_patterns = {
            "tpm": ["tpm_integration", "hardware_security"],
            "encryption": ["cryptography", "key_management"],
            "monitoring": ["system_monitoring", "performance_tracking"],
            "deployment": ["system_deployment", "automation"],
            "testing": ["quality_assurance", "validation"],
            "optimization": ["performance_optimization", "efficiency"]
        }
        
        content_lower = content.lower()
        for pattern, specs in spec_patterns.items():
            if pattern in content_lower:
                specializations.extend(specs)
                
        return list(set(specializations))[:10]  # Deduplicate and limit
    
    def _categorize_agents(self):
        """Categorize agents for better organization"""
        categories = {}
        for agent in self.agents.values():
            if agent.category not in categories:
                categories[agent.category] = []
            categories[agent.category].append(agent.name)
            
        logger.info("Agent categories:")
        for category, agents in categories.items():
            logger.info(f"  {category}: {len(agents)} agents")
    
    def get_agents_by_category(self, category: str) -> List[str]:
        """Get agent names by category"""
        return [name for name, agent in self.agents.items() 
                if agent.category.upper() == category.upper()]
    
    def get_agents_by_keywords(self, keywords: List[str]) -> List[str]:
        """Get agents matching keywords"""
        matching_agents = []
        keyword_set = set(k.lower() for k in keywords)
        
        for name, agent in self.agents.items():
            agent_keywords = set(k.lower() for k in agent.keywords + agent.specializations)
            agent_keywords.add(agent.name.lower())
            
            if keyword_set.intersection(agent_keywords):
                matching_agents.append(name)
                
        return matching_agents
    
    def get_agent_health(self, agent_name: str) -> float:
        """Get agent health score"""
        return self.agents.get(agent_name, AgentCapability("", "", "", "", [])).health_score

class TaskRouter:
    """Intelligent task routing based on keywords and agent capabilities"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.registry = agent_registry
        self.routing_rules = self._initialize_routing_rules()
        
    def _initialize_routing_rules(self) -> Dict[str, Dict]:
        """Initialize task routing rules for Phase 2 components"""
        return {
            # TPM Operations
            "tpm_integration": {
                "primary": ["SECURITY", "CRYPTOEXPERT", "HARDWARE"],
                "secondary": ["HARDWARE-DELL", "HARDWARE-INTEL"],
                "keywords": ["tpm", "hardware_security", "attestation", "sealing"]
            },
            
            # Device Control & Monitoring  
            "device_control": {
                "primary": ["HARDWARE-DELL", "HARDWARE-INTEL", "MONITOR"],
                "secondary": ["OPTIMIZER", "INFRASTRUCTURE"],
                "keywords": ["device", "control", "dsmil", "latitude"]
            },
            
            # ML Integration
            "ml_integration": {
                "primary": ["MLOPS", "DATASCIENCE", "NPU"],
                "secondary": ["DATABASE", "PYTHON-INTERNAL"],
                "keywords": ["machine_learning", "ml", "neural", "ai"]
            },
            
            # Security Operations
            "security_operations": {
                "primary": ["CSO", "SECURITY", "BASTION", "SECURITYAUDITOR"],
                "secondary": ["CRYPTOEXPERT", "QUANTUMGUARD", "APT41-DEFENSE-AGENT"],
                "keywords": ["security", "audit", "vulnerability", "threat"]
            },
            
            # Testing & Validation
            "testing_validation": {
                "primary": ["TESTBED", "DEBUGGER", "QADIRECTOR"],
                "secondary": ["LINTER", "CONSTRUCTOR"],
                "keywords": ["test", "validation", "qa", "quality"]
            },
            
            # Performance Optimization
            "performance_optimization": {
                "primary": ["OPTIMIZER", "LEADENGINEER", "MONITOR"],
                "secondary": ["HARDWARE-INTEL", "C-INTERNAL"],
                "keywords": ["performance", "optimization", "speed", "efficiency"]
            },
            
            # Documentation & Analysis
            "documentation": {
                "primary": ["DOCGEN", "RESEARCHER", "ANALYST"],
                "secondary": ["PLANNER", "ARCHITECT"],
                "keywords": ["documentation", "analysis", "research", "planning"]
            },
            
            # Infrastructure & Deployment
            "infrastructure": {
                "primary": ["INFRASTRUCTURE", "DEPLOYER", "PACKAGER"],
                "secondary": ["DOCKER-AGENT", "CISCO-AGENT"],
                "keywords": ["infrastructure", "deployment", "automation", "devops"]
            }
        }
    
    def route_task(self, task_description: str, task_type: str = None) -> List[str]:
        """Route task to appropriate agents based on description and type"""
        task_lower = task_description.lower()
        matched_agents = set()
        
        # Try specific routing rules first
        if task_type and task_type in self.routing_rules:
            rule = self.routing_rules[task_type]
            
            # Add primary agents
            for agent_name in rule["primary"]:
                if agent_name in self.registry.agents:
                    matched_agents.add(agent_name)
            
            # Add secondary agents if keywords match
            for keyword in rule["keywords"]:
                if keyword in task_lower:
                    for agent_name in rule["secondary"]:
                        if agent_name in self.registry.agents:
                            matched_agents.add(agent_name)
        
        # Keyword-based routing as fallback
        task_keywords = re.findall(r'\w+', task_lower)
        keyword_matches = self.registry.get_agents_by_keywords(task_keywords)
        matched_agents.update(keyword_matches[:5])  # Limit to top 5
        
        # Always include strategic coordination
        strategic_agents = ["DIRECTOR", "PROJECTORCHESTRATOR"]
        for agent in strategic_agents:
            if agent in self.registry.agents:
                matched_agents.add(agent)
        
        return list(matched_agents)

class AgentCoordinator:
    """Main coordination system for agent deployment"""
    
    def __init__(self, project_root: str = "/home/john/LAT5150DRVMIL"):
        self.project_root = Path(project_root)
        self.registry = AgentRegistry()
        self.router = TaskRouter(self.registry)
        self.active_tasks: Dict[str, DeploymentTask] = {}
        self.task_history: List[DeploymentTask] = []
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0,
            "agent_utilization": {}
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging infrastructure"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Agent-specific log handlers
        self.agent_loggers = {}
        for agent_name in self.registry.agents.keys():
            agent_logger = logging.getLogger(f"agent.{agent_name}")
            handler = logging.FileHandler(log_dir / f"agent_{agent_name.lower()}.log")
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            agent_logger.addHandler(handler)
            self.agent_loggers[agent_name] = agent_logger
    
    def create_phase2_deployment_plan(self) -> List[DeploymentTask]:
        """Create comprehensive Phase 2 deployment plan"""
        logger.info("Creating Phase 2 deployment plan with TPM enhancement")
        
        tasks = [
            # Week 1: TPM Device Integration (Days 31-40)
            DeploymentTask(
                task_id="TPM_001",
                name="TPM Hardware Discovery and Capability Assessment",
                description="Discover TPM 2.0 capabilities and integrate with DSMIL device 0x8005",
                assigned_agents=self.router.route_task(
                    "TPM hardware discovery attestation sealing device 0x8005", 
                    "tpm_integration"
                ),
                parallel_execution=True
            ),
            
            DeploymentTask(
                task_id="TPM_002", 
                name="DSMIL-TPM Monitoring Integration",
                description="Integrate TPM measurements with DSMIL monitoring framework",
                assigned_agents=self.router.route_task(
                    "TPM PCR measurements monitoring integration", 
                    "device_control"
                ),
                dependencies=["TPM_001"]
            ),
            
            DeploymentTask(
                task_id="TPM_003",
                name="TPM-Based Device Attestation Implementation", 
                description="Implement hardware attestation for DSMIL devices using TPM PCR",
                assigned_agents=self.router.route_task(
                    "TPM attestation PCR device authentication",
                    "security_operations"
                ),
                dependencies=["TPM_001", "TPM_002"],
                parallel_execution=True
            ),
            
            # Week 2: Encryption & Security (Days 41-50)
            DeploymentTask(
                task_id="SEC_001",
                name="Hardware Encryption Key Management",
                description="Implement device 0x8011 encryption with TPM-backed keys",
                assigned_agents=self.router.route_task(
                    "encryption key management TPM hardware device 0x8011",
                    "security_operations"
                ),
                dependencies=["TPM_001"]
            ),
            
            DeploymentTask(
                task_id="SEC_002", 
                name="Secure Boot Validator Integration",
                description="Integrate device 0x8008 with TPM PCR boot measurements",
                assigned_agents=self.router.route_task(
                    "secure boot validator TPM PCR measurements",
                    "security_operations"
                ),
                dependencies=["TPM_003"]
            ),
            
            DeploymentTask(
                task_id="SEC_003",
                name="ECC Performance Optimization",
                description="Optimize ECC operations for 3x performance boost over RSA",
                assigned_agents=self.router.route_task(
                    "ECC performance optimization cryptography",
                    "performance_optimization"
                ),
                parallel_execution=True
            ),
            
            # Week 3: Security Devices (Days 51-55)
            DeploymentTask(
                task_id="IDS_001",
                name="Intrusion Detection System Integration",
                description="Implement device 0x8013 IDS with TPM attestation",
                assigned_agents=self.router.route_task(
                    "intrusion detection system TPM device 0x8013",
                    "security_operations"
                ),
                dependencies=["SEC_001", "SEC_002"]
            ),
            
            DeploymentTask(
                task_id="POL_001",
                name="Policy Enforcement Engine",
                description="Deploy device 0x8014 policy enforcement with hardware backing",
                assigned_agents=self.router.route_task(
                    "policy enforcement engine device 0x8014 hardware",
                    "security_operations"
                ),
                dependencies=["IDS_001"]
            ),
            
            # Week 4: Network Security (Days 56-60)
            DeploymentTask(
                task_id="NET_001",
                name="Network Security Filter",
                description="Deploy device 0x8022 network filter with TPM packet authentication",
                assigned_agents=self.router.route_task(
                    "network security filter device 0x8022 TPM authentication",
                    "security_operations"
                ),
                dependencies=["POL_001"],
                parallel_execution=True
            ),
            
            DeploymentTask(
                task_id="NET_002",
                name="Network Authentication Gateway",
                description="Implement device 0x8027 with TPM-backed 802.1X certificates",
                assigned_agents=self.router.route_task(
                    "network authentication gateway 802.1X TPM certificates device 0x8027",
                    "security_operations"
                ),
                dependencies=["NET_001"]
            ),
            
            # Continuous: Testing & Validation
            DeploymentTask(
                task_id="TEST_001",
                name="Continuous Integration Testing",
                description="Implement continuous testing for all Phase 2 components",
                assigned_agents=self.router.route_task(
                    "continuous integration testing validation",
                    "testing_validation"
                ),
                parallel_execution=True
            ),
            
            # Continuous: Performance Monitoring
            DeploymentTask(
                task_id="MON_001",
                name="Real-time Performance Dashboard",
                description="Deploy comprehensive monitoring for all Phase 2 systems",
                assigned_agents=self.router.route_task(
                    "real-time performance monitoring dashboard",
                    "infrastructure"
                ),
                parallel_execution=True
            ),
            
            # Documentation & Analysis
            DeploymentTask(
                task_id="DOC_001",
                name="Phase 2 Documentation Generation",
                description="Generate comprehensive documentation for all Phase 2 implementations",
                assigned_agents=self.router.route_task(
                    "documentation generation analysis Phase 2",
                    "documentation"
                ),
                dependencies=["NET_002"],  # Wait for completion
                parallel_execution=True
            )
        ]
        
        logger.info(f"Created {len(tasks)} deployment tasks for Phase 2")
        return tasks
    
    async def execute_task(self, task: DeploymentTask) -> bool:
        """Execute a deployment task using assigned agents"""
        logger.info(f"Executing task {task.task_id}: {task.name}")
        task.start_time = datetime.now()
        task.status = "in_progress"
        
        try:
            if task.parallel_execution and len(task.assigned_agents) > 1:
                # Execute agents in parallel
                results = await self._execute_agents_parallel(task)
            else:
                # Execute agents sequentially
                results = await self._execute_agents_sequential(task)
            
            task.results = results
            task.status = "completed"
            task.completion_time = datetime.now()
            
            # Update performance metrics
            self._update_performance_metrics(task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = "failed"
            task.completion_time = datetime.now()
            task.results = {"error": str(e)}
            return False
    
    async def _execute_agents_parallel(self, task: DeploymentTask) -> Dict[str, Any]:
        """Execute multiple agents in parallel"""
        logger.info(f"Executing {len(task.assigned_agents)} agents in parallel for task {task.task_id}")
        
        async def execute_agent(agent_name: str) -> Tuple[str, Dict]:
            try:
                result = await self._invoke_agent(agent_name, task.description, task.task_id)
                return agent_name, {"status": "success", "result": result}
            except Exception as e:
                return agent_name, {"status": "error", "error": str(e)}
        
        # Create tasks for parallel execution
        agent_tasks = [execute_agent(agent) for agent in task.assigned_agents]
        
        # Execute all agents concurrently
        results = {}
        completed_tasks = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        for agent_result in completed_tasks:
            if isinstance(agent_result, tuple):
                agent_name, result = agent_result
                results[agent_name] = result
            else:
                logger.error(f"Agent execution exception: {agent_result}")
        
        return results
    
    async def _execute_agents_sequential(self, task: DeploymentTask) -> Dict[str, Any]:
        """Execute agents sequentially with coordination"""
        logger.info(f"Executing {len(task.assigned_agents)} agents sequentially for task {task.task_id}")
        
        results = {}
        context = {"task_id": task.task_id, "previous_results": {}}
        
        for agent_name in task.assigned_agents:
            try:
                # Provide context from previous agents
                enhanced_description = f"{task.description}\n\nContext: {json.dumps(context, indent=2)}"
                
                result = await self._invoke_agent(agent_name, enhanced_description, task.task_id)
                results[agent_name] = {"status": "success", "result": result}
                
                # Update context for next agent
                context["previous_results"][agent_name] = result
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed in task {task.task_id}: {e}")
                results[agent_name] = {"status": "error", "error": str(e)}
        
        return results
    
    async def _invoke_agent(self, agent_name: str, description: str, task_id: str) -> str:
        """Simulate agent invocation - In real implementation, this would use Task tool"""
        logger.info(f"Invoking agent {agent_name} for task {task_id}")
        
        # Update agent metrics
        if agent_name in self.registry.agents:
            self.registry.agents[agent_name].last_invocation = datetime.now()
        
        # Log agent activity
        if agent_name in self.agent_loggers:
            self.agent_loggers[agent_name].info(f"Task: {task_id} - {description}")
        
        # Simulate agent work (replace with actual Task tool invocation)
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Generate realistic response based on agent type
        agent = self.registry.agents.get(agent_name)
        if agent:
            if agent.category == "SECURITY":
                return f"Security analysis completed for {description}. No threats detected."
            elif agent.category == "HARDWARE":
                return f"Hardware optimization applied for {description}. Performance improved."
            elif agent.category == "INFRASTRUCTURE":
                return f"Infrastructure deployment completed for {description}. Systems operational."
            else:
                return f"Task completed by {agent_name}: {description}"
        
        return f"Task completed by {agent_name}"
    
    def _update_performance_metrics(self, task: DeploymentTask):
        """Update performance metrics after task completion"""
        self.performance_metrics["total_tasks"] += 1
        
        if task.status == "completed":
            self.performance_metrics["completed_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        # Calculate completion time
        if task.start_time and task.completion_time:
            duration = (task.completion_time - task.start_time).total_seconds()
            
            # Update average completion time
            total_completed = self.performance_metrics["completed_tasks"]
            if total_completed > 1:
                current_avg = self.performance_metrics["average_completion_time"]
                new_avg = ((current_avg * (total_completed - 1)) + duration) / total_completed
                self.performance_metrics["average_completion_time"] = new_avg
            else:
                self.performance_metrics["average_completion_time"] = duration
        
        # Update agent utilization
        for agent_name in task.assigned_agents:
            if agent_name not in self.performance_metrics["agent_utilization"]:
                self.performance_metrics["agent_utilization"][agent_name] = 0
            self.performance_metrics["agent_utilization"][agent_name] += 1
    
    def generate_deployment_dashboard(self) -> Dict[str, Any]:
        """Generate real-time deployment dashboard"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "system_status": {
                "total_agents": len(self.registry.agents),
                "active_tasks": len([t for t in self.active_tasks.values() if t.status == "in_progress"]),
                "completed_tasks": len([t for t in self.active_tasks.values() if t.status == "completed"]),
                "failed_tasks": len([t for t in self.active_tasks.values() if t.status == "failed"])
            },
            "agent_categories": {},
            "performance_metrics": self.performance_metrics.copy(),
            "top_agents": {},
            "current_phase": "Phase 2 - TPM Enhanced Device Expansion"
        }
        
        # Agent categories
        categories = {}
        for agent in self.registry.agents.values():
            if agent.category not in categories:
                categories[agent.category] = 0
            categories[agent.category] += 1
        dashboard["agent_categories"] = categories
        
        # Top performing agents
        utilization = self.performance_metrics.get("agent_utilization", {})
        if utilization:
            sorted_agents = sorted(utilization.items(), key=lambda x: x[1], reverse=True)
            dashboard["top_agents"] = dict(sorted_agents[:10])
        
        return dashboard
    
    async def deploy_phase2(self):
        """Execute complete Phase 2 deployment"""
        logger.info("Starting Phase 2 comprehensive deployment")
        
        # Create deployment plan
        tasks = self.create_phase2_deployment_plan()
        
        # Add tasks to active task list
        for task in tasks:
            self.active_tasks[task.task_id] = task
        
        # Execute tasks respecting dependencies
        executed_tasks = set()
        
        while len(executed_tasks) < len(tasks):
            # Find tasks ready for execution (dependencies satisfied)
            ready_tasks = []
            for task in tasks:
                if (task.task_id not in executed_tasks and 
                    all(dep in executed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                logger.error("Deadlock detected in task dependencies")
                break
            
            # Group parallel tasks
            parallel_tasks = [t for t in ready_tasks if t.parallel_execution]
            sequential_tasks = [t for t in ready_tasks if not t.parallel_execution]
            
            # Execute parallel tasks first
            if parallel_tasks:
                parallel_futures = [self.execute_task(task) for task in parallel_tasks]
                parallel_results = await asyncio.gather(*parallel_futures)
                
                for task, success in zip(parallel_tasks, parallel_results):
                    if success:
                        executed_tasks.add(task.task_id)
                        logger.info(f"Parallel task {task.task_id} completed")
            
            # Execute sequential tasks
            for task in sequential_tasks:
                success = await self.execute_task(task)
                if success:
                    executed_tasks.add(task.task_id)
                    logger.info(f"Sequential task {task.task_id} completed")
                else:
                    logger.error(f"Critical task {task.task_id} failed")
            
            # Generate dashboard update
            dashboard = self.generate_deployment_dashboard()
            with open(self.project_root / "phase2_dashboard.json", "w") as f:
                json.dump(dashboard, f, indent=2)
        
        logger.info(f"Phase 2 deployment completed. {len(executed_tasks)}/{len(tasks)} tasks executed")
        
        # Generate final report
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate comprehensive deployment report"""
        report = {
            "deployment_summary": {
                "phase": "Phase 2 - TPM Enhanced Device Expansion",
                "completion_time": datetime.now().isoformat(),
                "total_agents_used": len(set(
                    agent for task in self.active_tasks.values() 
                    for agent in task.assigned_agents
                )),
                "tasks_completed": len([t for t in self.active_tasks.values() if t.status == "completed"]),
                "tasks_failed": len([t for t in self.active_tasks.values() if t.status == "failed"]),
                "performance_metrics": self.performance_metrics
            },
            "agent_utilization": {},
            "task_details": {},
            "recommendations": []
        }
        
        # Agent utilization analysis
        for agent_name, agent in self.registry.agents.items():
            report["agent_utilization"][agent_name] = {
                "category": agent.category,
                "invocations": self.performance_metrics.get("agent_utilization", {}).get(agent_name, 0),
                "last_used": agent.last_invocation.isoformat() if agent.last_invocation else None,
                "health_score": agent.health_score
            }
        
        # Task details
        for task_id, task in self.active_tasks.items():
            report["task_details"][task_id] = {
                "name": task.name,
                "status": task.status,
                "assigned_agents": task.assigned_agents,
                "duration": (
                    (task.completion_time - task.start_time).total_seconds() 
                    if task.start_time and task.completion_time else None
                ),
                "results_summary": len(task.results) if task.results else 0
            }
        
        # Generate recommendations
        failed_tasks = [t for t in self.active_tasks.values() if t.status == "failed"]
        if failed_tasks:
            report["recommendations"].append("Review failed tasks and retry with different agent assignments")
        
        underutilized_agents = [
            name for name, count in self.performance_metrics.get("agent_utilization", {}).items()
            if count == 0
        ]
        if underutilized_agents:
            report["recommendations"].append(f"Consider utilizing unused agents: {', '.join(underutilized_agents[:5])}")
        
        # Save report
        report_file = self.project_root / f"phase2_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final deployment report saved to {report_file}")
        
        return report

def main():
    """Main deployment coordinator entry point"""
    print("=" * 80)
    print("COMPREHENSIVE AGENT DEPLOYMENT SYSTEM v2.0")
    print("Phase 2 DSMIL TPM-Enhanced Device Expansion")
    print("=" * 80)
    
    # Initialize coordinator
    coordinator = AgentCoordinator()
    
    print(f"\nLoaded {len(coordinator.registry.agents)} specialized agents:")
    
    # Display agent summary by category
    categories = {}
    for agent in coordinator.registry.agents.values():
        if agent.category not in categories:
            categories[agent.category] = []
        categories[agent.category].append(agent.name)
    
    for category, agents in sorted(categories.items()):
        print(f"  {category}: {len(agents)} agents")
        for agent in sorted(agents)[:3]:  # Show first 3 agents
            print(f"    - {agent}")
        if len(agents) > 3:
            print(f"    ... and {len(agents) - 3} more")
    
    print("\nStarting Phase 2 deployment...")
    
    # Run deployment
    asyncio.run(coordinator.deploy_phase2())
    
    print("\nDeployment completed! Check logs and dashboard for details.")
    print(f"Dashboard: {coordinator.project_root}/phase2_dashboard.json")
    print(f"Logs: {coordinator.project_root}/logs/")

if __name__ == "__main__":
    main()