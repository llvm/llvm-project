#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Advanced Self-Awareness Engine
True AI introspection with dynamic capability discovery and reasoning
"""

import os
import sys
import ast
import json
import inspect
import importlib
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import sqlite3
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SelfAwareness: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemResource:
    """Discovered system resource"""
    resource_type: str  # model, hardware, service, tool
    name: str
    location: str
    status: str  # available, unavailable, degraded
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_checked: str = ""


@dataclass
class DiscoveredCapability:
    """Dynamically discovered capability through code introspection"""
    id: str
    name: str
    source_file: str
    source_line: int
    function_name: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    dependencies: List[str]
    category: str
    confidence: float  # How confident we are this is a real capability


@dataclass
class SystemState:
    """Current system state"""
    uptime_seconds: float
    active_capabilities: List[str]
    available_resources: List[SystemResource]
    recent_interactions: List[Dict[str, Any]]
    current_load: Dict[str, float]  # cpu, memory, etc.
    errors_last_hour: int
    last_state_update: str


@dataclass
class ReasoningContext:
    """Context for reasoning about capabilities"""
    user_query: str
    available_capabilities: List[str]
    system_state: SystemState
    past_successful_actions: List[str]
    constraints: Dict[str, Any]


class SelfAwarenessEngine:
    """
    Advanced self-awareness system that:
    - Introspects codebase to discover capabilities
    - Understands system state and resources
    - Reasons about what it can/cannot do
    - Maintains memory of interactions
    - Learns from successes/failures
    """

    def __init__(self, workspace_path: str, state_db_path: str = "/opt/lat5150/state/self_awareness.db"):
        self.workspace_path = Path(workspace_path)
        self.state_db_path = Path(state_db_path)
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize state database
        self._init_state_db()

        # Discovered capabilities (dynamic)
        self.discovered_capabilities: Dict[str, DiscoveredCapability] = {}

        # Available resources (models, hardware, services)
        self.resources: Dict[str, SystemResource] = {}

        # System state
        self.state = SystemState(
            uptime_seconds=0,
            active_capabilities=[],
            available_resources=[],
            recent_interactions=[],
            current_load={},
            errors_last_hour=0,
            last_state_update=datetime.utcnow().isoformat() + "Z"
        )

        # Interaction memory
        self.interaction_history: List[Dict[str, Any]] = []

        # Start time
        self.start_time = datetime.utcnow()

        logger.info("Self-Awareness Engine initialized")

    def _init_state_db(self):
        """Initialize state persistence database"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()

        # Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT,
                matched_capability TEXT,
                success BOOLEAN,
                execution_time_ms REAL,
                error TEXT,
                context TEXT
            )
        """)

        # Capabilities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                name TEXT,
                source_file TEXT,
                function_name TEXT,
                category TEXT,
                times_used INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                avg_execution_time_ms REAL DEFAULT 0,
                last_used TEXT
            )
        """)

        # Resources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                resource_type TEXT,
                name TEXT,
                location TEXT,
                status TEXT,
                last_checked TEXT,
                metadata TEXT
            )
        """)

        # System state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                timestamp TEXT PRIMARY KEY,
                uptime_seconds REAL,
                active_capabilities INTEGER,
                available_resources INTEGER,
                cpu_percent REAL,
                memory_percent REAL,
                errors_count INTEGER
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"State database initialized: {self.state_db_path}")

    def discover_capabilities(self) -> Dict[str, DiscoveredCapability]:
        """
        Discover capabilities by introspecting codebase

        Searches for:
        - Functions with specific patterns
        - Classes with execute/run methods
        - Registered handlers
        - API endpoints
        """
        logger.info("Starting capability discovery...")

        discovered = {}

        # Scan specific directories
        scan_dirs = [
            self.workspace_path / "01-source",
            self.workspace_path / "03-web-interface",
            self.workspace_path / "02-hardware-integration"
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue

            for py_file in scan_dir.rglob("*.py"):
                try:
                    capabilities = self._analyze_python_file(py_file)
                    for cap in capabilities:
                        discovered[cap.id] = cap
                except Exception as e:
                    logger.debug(f"Error analyzing {py_file}: {e}")

        self.discovered_capabilities = discovered

        # Persist to database
        self._persist_capabilities(discovered)

        logger.info(f"Discovered {len(discovered)} capabilities")
        return discovered

    def _analyze_python_file(self, file_path: Path) -> List[DiscoveredCapability]:
        """Analyze a Python file to discover capabilities"""
        capabilities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Look for functions that look like capabilities
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if this looks like a capability
                    if self._is_capability_function(node):
                        cap = self._extract_capability(node, file_path, source)
                        if cap:
                            capabilities.append(cap)

                elif isinstance(node, ast.ClassDef):
                    # Check for classes with execute/run methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name in ['execute', 'run', 'process']:
                            cap = self._extract_capability(item, file_path, source, class_name=node.name)
                            if cap:
                                capabilities.append(cap)

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return capabilities

    def _is_capability_function(self, node: ast.FunctionDef) -> bool:
        """Determine if a function represents a capability"""
        # Skip private functions
        if node.name.startswith('_'):
            return False

        # Look for capability indicators
        indicators = [
            'execute', 'run', 'process', 'handle', 'scan',
            'analyze', 'detect', 'find', 'search', 'invoke',
            'verify', 'check', 'monitor', 'discover'
        ]

        name_lower = node.name.lower()
        return any(indicator in name_lower for indicator in indicators)

    def _extract_capability(self, node: ast.FunctionDef, file_path: Path,
                           source: str, class_name: Optional[str] = None) -> Optional[DiscoveredCapability]:
        """Extract capability details from AST node"""
        try:
            # Get docstring
            docstring = ast.get_docstring(node)

            # Extract parameters
            parameters = []
            for arg in node.args.args:
                param = {
                    "name": arg.arg,
                    "type": self._get_annotation(arg.annotation) if arg.annotation else "Any",
                    "required": True
                }
                parameters.append(param)

            # Get return type
            return_type = self._get_annotation(node.returns) if node.returns else None

            # Determine category
            category = self._categorize_capability(node.name, docstring)

            # Generate ID
            full_name = f"{class_name}.{node.name}" if class_name else node.name
            cap_id = hashlib.md5(f"{file_path}:{full_name}".encode()).hexdigest()[:16]

            # Get relative path
            try:
                rel_path = file_path.relative_to(self.workspace_path)
            except ValueError:
                rel_path = file_path

            return DiscoveredCapability(
                id=cap_id,
                name=full_name,
                source_file=str(rel_path),
                source_line=node.lineno,
                function_name=node.name,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                dependencies=self._extract_dependencies(node, source),
                category=category,
                confidence=0.8  # Base confidence
            )

        except Exception as e:
            logger.debug(f"Error extracting capability from {node.name}: {e}")
            return None

    def _get_annotation(self, annotation) -> str:
        """Get string representation of type annotation"""
        if annotation is None:
            return "Any"
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant):
            return str(annotation.value)
        return "Any"

    def _categorize_capability(self, name: str, docstring: Optional[str]) -> str:
        """Categorize capability based on name and docstring"""
        name_lower = name.lower()
        doc_lower = (docstring or "").lower()

        combined = name_lower + " " + doc_lower

        categories = {
            "code_understanding": ["find", "search", "locate", "symbol", "reference", "parse", "analyze_code"],
            "agent_execution": ["agent", "invoke", "execute", "run", "container"],
            "model_inference": ["model", "generate", "complete", "infer", "predict"],
            "hardware_recon": ["hardware", "device", "scan", "detect", "dsmil", "pci"],
            "security_audit": ["audit", "verify", "check", "security", "hash", "chain"],
            "system_control": ["control", "manage", "configure", "tempest", "mode"],
            "data_processing": ["process", "transform", "convert", "parse"],
            "network": ["network", "request", "fetch", "http", "api"],
            "file_operations": ["file", "read", "write", "edit"],
            "monitoring": ["monitor", "health", "status", "metric"]
        }

        for category, keywords in categories.items():
            if any(keyword in combined for keyword in keywords):
                return category

        return "general"

    def _extract_dependencies(self, node: ast.FunctionDef, source: str) -> List[str]:
        """Extract dependencies from function"""
        dependencies = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    dependencies.add(alias.name)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    dependencies.add(child.module)

        return list(dependencies)

    def discover_resources(self) -> Dict[str, SystemResource]:
        """
        Discover available system resources:
        - Local AI models (Ollama)
        - Hardware devices (DSMIL)
        - Running services
        - Available tools
        """
        logger.info("Discovering system resources...")

        resources = {}

        # Discover Ollama models
        resources.update(self._discover_ollama_models())

        # Discover DSMIL hardware
        resources.update(self._discover_dsmil_hardware())

        # Discover running services
        resources.update(self._discover_services())

        # Discover available tools
        resources.update(self._discover_tools())

        self.resources = resources
        self._persist_resources(resources)

        logger.info(f"Discovered {len(resources)} resources")
        return resources

    def _discover_ollama_models(self) -> Dict[str, SystemResource]:
        """Discover available Ollama models"""
        resources = {}

        try:
            import aiohttp
            import asyncio

            async def check_ollama():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get('http://localhost:11434/api/tags', timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return data.get('models', [])
                except:
                    return []

            loop = asyncio.get_event_loop()
            models = loop.run_until_complete(check_ollama())

            for model in models:
                resource_id = f"model_{model['name'].replace(':', '_')}"
                resources[resource_id] = SystemResource(
                    resource_type="model",
                    name=model['name'],
                    location="http://localhost:11434",
                    status="available",
                    metadata={
                        "size": model.get('size', 0),
                        "modified": model.get('modified_at', '')
                    },
                    last_checked=datetime.utcnow().isoformat() + "Z"
                )

        except Exception as e:
            logger.debug(f"Error discovering Ollama models: {e}")

        return resources

    def _discover_dsmil_hardware(self) -> Dict[str, SystemResource]:
        """Discover DSMIL hardware devices"""
        resources = {}

        try:
            # Check if DSMIL module exists
            dsmil_path = self.workspace_path / "02-hardware-integration" / "dsmil-kernel-module"
            if dsmil_path.exists():
                resources["dsmil_module"] = SystemResource(
                    resource_type="hardware",
                    name="DSMIL Kernel Module",
                    location=str(dsmil_path),
                    status="available",
                    metadata={"device_count": 84},
                    last_checked=datetime.utcnow().isoformat() + "Z"
                )

        except Exception as e:
            logger.debug(f"Error discovering DSMIL hardware: {e}")

        return resources

    def _discover_services(self) -> Dict[str, SystemResource]:
        """Discover running services"""
        resources = {}

        # Check Docker/Podman
        for runtime in ['docker', 'podman']:
            try:
                import subprocess
                result = subprocess.run([runtime, '--version'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    resources[f"service_{runtime}"] = SystemResource(
                        resource_type="service",
                        name=runtime.title(),
                        location=f"/usr/bin/{runtime}",
                        status="available",
                        metadata={"version": result.stdout.strip()},
                        last_checked=datetime.utcnow().isoformat() + "Z"
                    )
            except:
                pass

        return resources

    def _discover_tools(self) -> Dict[str, SystemResource]:
        """Discover available command-line tools"""
        resources = {}

        tools = ['git', 'curl', 'jq', 'python3', 'gcc', 'make']

        for tool in tools:
            try:
                import subprocess
                result = subprocess.run(['which', tool], capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    resources[f"tool_{tool}"] = SystemResource(
                        resource_type="tool",
                        name=tool,
                        location=result.stdout.strip(),
                        status="available",
                        metadata={},
                        last_checked=datetime.utcnow().isoformat() + "Z"
                    )
            except:
                pass

        return resources

    def update_system_state(self):
        """Update current system state"""
        try:
            import psutil

            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            self.state = SystemState(
                uptime_seconds=uptime,
                active_capabilities=list(self.discovered_capabilities.keys()),
                available_resources=list(self.resources.values()),
                recent_interactions=self.interaction_history[-10:],
                current_load={
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent
                },
                errors_last_hour=self._count_recent_errors(),
                last_state_update=datetime.utcnow().isoformat() + "Z"
            )

            self._persist_state()

        except Exception as e:
            logger.debug(f"Error updating system state: {e}")

    def reason_about_capability(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Reason about whether we can handle a query

        Returns:
        - can_handle: bool
        - confidence: float
        - matched_capabilities: List[str]
        - reasoning: str (explanation)
        - requirements: List[str] (what's needed)
        - alternatives: List[str] (if can't handle directly)
        """
        query_lower = context.user_query.lower()

        # Check if we have matching capabilities
        matched_caps = []
        for cap_id, cap in self.discovered_capabilities.items():
            # Simple matching for now
            if cap.function_name.lower() in query_lower:
                matched_caps.append(cap_id)
            elif cap.category in query_lower:
                matched_caps.append(cap_id)

        # Check if we have required resources
        required_resources = self._infer_required_resources(context.user_query)
        available_resources = [r for r in self.resources.values() if r.status == "available"]

        has_resources = all(
            any(r.resource_type == req_type for r in available_resources)
            for req_type in required_resources
        )

        # Generate reasoning
        if matched_caps and has_resources:
            reasoning = f"I can handle this query. I have {len(matched_caps)} matching capabilities and all required resources are available."
            can_handle = True
            confidence = 0.9
        elif matched_caps:
            reasoning = f"I have capabilities to handle this, but some resources may be unavailable: {', '.join(required_resources)}"
            can_handle = True
            confidence = 0.6
        else:
            reasoning = "I don't have a direct capability for this query, but I can attempt to break it down into smaller tasks."
            can_handle = False
            confidence = 0.3

        # Suggest alternatives
        alternatives = []
        if not can_handle:
            # Find similar capabilities
            for cap_id, cap in self.discovered_capabilities.items():
                if cap.category in query_lower or any(word in cap.name.lower() for word in query_lower.split()):
                    alternatives.append(cap.name)

        return {
            "can_handle": can_handle,
            "confidence": confidence,
            "matched_capabilities": matched_caps,
            "reasoning": reasoning,
            "requirements": required_resources,
            "alternatives": alternatives[:5]  # Top 5
        }

    def _infer_required_resources(self, query: str) -> List[str]:
        """Infer what resources are needed for a query"""
        resources = []
        query_lower = query.lower()

        if any(word in query_lower for word in ['model', 'generate', 'analyze', 'understand']):
            resources.append('model')

        if any(word in query_lower for word in ['hardware', 'device', 'scan']):
            resources.append('hardware')

        if any(word in query_lower for word in ['container', 'docker', 'agent']):
            resources.append('service')

        return resources

    def record_interaction(self, query: str, matched_capability: Optional[str],
                          success: bool, execution_time_ms: float, error: Optional[str] = None):
        """Record an interaction for learning"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "matched_capability": matched_capability,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error": error
        }

        self.interaction_history.append(interaction)

        # Persist to database
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (timestamp, query, matched_capability, success, execution_time_ms, error, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction["timestamp"],
            query,
            matched_capability,
            success,
            execution_time_ms,
            error,
            json.dumps({})
        ))
        conn.commit()
        conn.close()

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive self-awareness report"""
        self.update_system_state()

        # Capability statistics
        cap_stats = {
            "total": len(self.discovered_capabilities),
            "by_category": {},
            "by_file": {}
        }

        for cap in self.discovered_capabilities.values():
            cap_stats["by_category"][cap.category] = cap_stats["by_category"].get(cap.category, 0) + 1
            cap_stats["by_file"][cap.source_file] = cap_stats["by_file"].get(cap.source_file, 0) + 1

        # Resource statistics
        resource_stats = {
            "total": len(self.resources),
            "by_type": {},
            "available": sum(1 for r in self.resources.values() if r.status == "available")
        }

        for res in self.resources.values():
            resource_stats["by_type"][res.resource_type] = resource_stats["by_type"].get(res.resource_type, 0) + 1

        # Performance statistics
        perf_stats = self._get_performance_stats()

        return {
            "system_name": "LAT5150 DRVMIL Tactical AI Sub-Engine",
            "self_awareness_level": "advanced",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": self.state.uptime_seconds,
            "capabilities": {
                "discovered": cap_stats,
                "top_used": self._get_top_used_capabilities(10),
                "examples": self._generate_capability_examples()
            },
            "resources": {
                "discovered": resource_stats,
                "details": [asdict(r) for r in list(self.resources.values())[:20]]
            },
            "system_state": asdict(self.state),
            "performance": perf_stats,
            "learning": {
                "total_interactions": len(self.interaction_history),
                "success_rate": self._calculate_success_rate(),
                "recent_errors": self._get_recent_errors(5)
            },
            "reasoning_ability": {
                "can_introspect": True,
                "can_reason_about_queries": True,
                "can_learn_from_feedback": True,
                "can_discover_capabilities": True
            }
        }

    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "avg_response_time_ms": self._calculate_avg_response_time(),
            "cpu_usage_percent": self.state.current_load.get("cpu_percent", 0),
            "memory_usage_percent": self.state.current_load.get("memory_percent", 0),
            "errors_last_hour": self.state.errors_last_hour
        }

    def _get_top_used_capabilities(self, limit: int) -> List[Dict[str, Any]]:
        """Get most used capabilities"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, times_used, success_rate, avg_execution_time_ms
            FROM capabilities
            ORDER BY times_used DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "name": row[0],
                "times_used": row[1],
                "success_rate": row[2],
                "avg_execution_time_ms": row[3]
            })

        conn.close()
        return results

    def _generate_capability_examples(self) -> List[str]:
        """Generate example queries for capabilities"""
        examples = []

        categories_seen = set()
        for cap in list(self.discovered_capabilities.values())[:10]:
            if cap.category not in categories_seen:
                categories_seen.add(cap.category)

                # Generate natural language example
                if cap.category == "code_understanding":
                    examples.append(f"Find the {cap.function_name} in the codebase")
                elif cap.category == "agent_execution":
                    examples.append(f"Run {cap.function_name} agent")
                elif cap.category == "hardware_recon":
                    examples.append(f"Scan for devices using {cap.function_name}")
                else:
                    examples.append(f"Execute {cap.function_name}")

        return examples

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.interaction_history:
            return 1.0

        successes = sum(1 for i in self.interaction_history if i.get("success", False))
        return successes / len(self.interaction_history)

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.interaction_history:
            return 0.0

        times = [i.get("execution_time_ms", 0) for i in self.interaction_history]
        return sum(times) / len(times) if times else 0.0

    def _get_recent_errors(self, limit: int) -> List[Dict[str, str]]:
        """Get recent errors"""
        errors = [
            {"timestamp": i["timestamp"], "error": i["error"]}
            for i in self.interaction_history
            if not i.get("success", False) and i.get("error")
        ]
        return errors[-limit:]

    def _count_recent_errors(self) -> int:
        """Count errors in the last hour"""
        one_hour_ago = datetime.utcnow().timestamp() - 3600

        count = 0
        for interaction in self.interaction_history:
            try:
                ts = datetime.fromisoformat(interaction["timestamp"].replace("Z", "")).timestamp()
                if ts > one_hour_ago and not interaction.get("success", False):
                    count += 1
            except:
                pass

        return count

    def _persist_capabilities(self, capabilities: Dict[str, DiscoveredCapability]):
        """Persist capabilities to database"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()

        for cap in capabilities.values():
            cursor.execute("""
                INSERT OR REPLACE INTO capabilities (id, name, source_file, function_name, category)
                VALUES (?, ?, ?, ?, ?)
            """, (cap.id, cap.name, cap.source_file, cap.function_name, cap.category))

        conn.commit()
        conn.close()

    def _persist_resources(self, resources: Dict[str, SystemResource]):
        """Persist resources to database"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()

        for resource in resources.values():
            cursor.execute("""
                INSERT OR REPLACE INTO resources (id, resource_type, name, location, status, last_checked, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{resource.resource_type}_{resource.name}",
                resource.resource_type,
                resource.name,
                resource.location,
                resource.status,
                resource.last_checked,
                json.dumps(resource.metadata)
            ))

        conn.commit()
        conn.close()

    def _persist_state(self):
        """Persist system state"""
        conn = sqlite3.connect(self.state_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO system_state (timestamp, uptime_seconds, active_capabilities,
                                     available_resources, cpu_percent, memory_percent, errors_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.state.last_state_update,
            self.state.uptime_seconds,
            len(self.state.active_capabilities),
            len(self.state.available_resources),
            self.state.current_load.get("cpu_percent", 0),
            self.state.current_load.get("memory_percent", 0),
            self.state.errors_last_hour
        ))

        conn.commit()
        conn.close()


# Test
async def main():
    engine = SelfAwarenessEngine(workspace_path="/home/user/LAT5150DRVMIL")

    print("Discovering capabilities...")
    caps = engine.discover_capabilities()
    print(f"Found {len(caps)} capabilities")

    print("\nDiscovering resources...")
    resources = engine.discover_resources()
    print(f"Found {len(resources)} resources")

    print("\nGenerating self-awareness report...")
    report = engine.get_comprehensive_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
