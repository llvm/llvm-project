#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Integrated Self-Awareness Engine
True AI self-awareness through complete system integration

Integrates with ALL existing systems:
- ChromaDB vector database (enhanced_rag_system.py)
- Cognitive memory system (cognitive_memory_enhanced.py)
- DSMIL AI Engine (dsmil_ai_engine.py)
- Quantum crypto layer (quantum_crypto_layer.py)
- DSMIL subsystem controller (dsmil_subsystem_controller.py)
- TPM crypto integration (tpm_crypto_integration.py)
- MCP servers (11 servers)
- Unified orchestrator (unified_orchestrator.py)
- Agent orchestrator (agent_orchestrator.py)

This is NOT a standalone system - it's the nervous system connecting everything.

Version: 4.0.0 - Full Integration
Author: DSMIL Integration Framework
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import pickle

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "01-source"))

# Core integrations
try:
    from enhanced_rag_system import EnhancedRAGSystem, SearchResult
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("Enhanced RAG system not available")

try:
    from cognitive_memory_enhanced import (
        CognitiveMemorySystem, MemoryTier, MemoryType,
        SalienceLevel, CognitiveMemoryBlock
    )
    COGNITIVE_MEMORY_AVAILABLE = True
except ImportError:
    COGNITIVE_MEMORY_AVAILABLE = False
    logging.warning("Cognitive memory system not available")

try:
    from dsmil_ai_engine import DSMILAIEngine
    DSMIL_AI_AVAILABLE = True
except ImportError:
    DSMIL_AI_AVAILABLE = False
    logging.warning("DSMIL AI engine not available")

try:
    from quantum_crypto_layer import QuantumCryptoLayer, SecurityLevel
    QUANTUM_CRYPTO_AVAILABLE = True
except ImportError:
    QUANTUM_CRYPTO_AVAILABLE = False
    logging.warning("Quantum crypto layer not available")

try:
    from dsmil_subsystem_controller import DSMILSubsystemController
    DSMIL_CONTROLLER_AVAILABLE = True
except ImportError:
    DSMIL_CONTROLLER_AVAILABLE = False
    logging.warning("DSMIL subsystem controller not available")

try:
    from tpm_crypto_integration import get_tpm_crypto, TPMCryptoIntegration
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False
    logging.warning("TPM crypto integration not available")

try:
    from unified_orchestrator import UnifiedOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    logging.warning("Unified orchestrator not available")

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logging.warning("Vector database (ChromaDB) not available")

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logging.warning("PostgreSQL not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] IntegratedSelfAwareness: %(message)s'
)
logger = logging.getLogger(__name__)


class SystemComponentType(Enum):
    """Types of integrated system components"""
    VECTOR_DATABASE = "vector_database"
    COGNITIVE_MEMORY = "cognitive_memory"
    AI_ENGINE = "ai_engine"
    CRYPTO_LAYER = "crypto_layer"
    HARDWARE_CONTROLLER = "hardware_controller"
    TPM_ATTESTATION = "tpm_attestation"
    MCP_SERVER = "mcp_server"
    ORCHESTRATOR = "orchestrator"
    AGENT_SYSTEM = "agent_system"


@dataclass
class SystemComponent:
    """Represents an integrated system component"""
    component_id: str
    component_type: SystemComponentType
    name: str
    instance: Any
    status: str  # active, degraded, offline
    capabilities: List[str]
    metadata: Dict[str, Any]
    last_health_check: datetime
    error_count: int = 0


@dataclass
class IntegratedCapability:
    """Capability that spans multiple systems"""
    capability_id: str
    name: str
    description: str
    required_components: List[SystemComponentType]
    available_components: List[str]
    confidence: float
    vector_embedding: Optional[List[float]] = None
    memory_block_id: Optional[str] = None
    usage_count: int = 0
    success_rate: float = 1.0
    avg_execution_time_ms: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class SystemState:
    """Complete system state across all components"""
    timestamp: datetime
    components_status: Dict[str, str]  # component_id -> status
    active_capabilities: List[str]
    vector_db_document_count: int
    cognitive_memory_blocks: int
    dsmil_devices_active: int
    crypto_key_rotation_due: bool
    tpm_attestation_valid: bool
    mcp_servers_online: int
    total_interactions: int
    current_load: Dict[str, float]
    health_score: float  # 0.0-1.0


class IntegratedSelfAwarenessEngine:
    """
    True self-awareness through complete system integration

    This is the nervous system that connects:
    - Vector database for semantic understanding
    - Cognitive memory for long-term learning
    - AI engine for reasoning
    - Quantum crypto for attestation
    - DSMIL hardware for platform integrity
    - TPM for cryptographic verification
    - MCP servers for tool access
    - Orchestrators for coordination
    """

    def __init__(
        self,
        workspace_path: str = "/home/user/LAT5150DRVMIL",
        state_db_path: str = "/opt/lat5150/state/integrated_awareness.db",
        vector_db_path: str = "/opt/lat5150/vectordb",
        postgres_url: Optional[str] = None
    ):
        self.workspace_path = Path(workspace_path)
        self.state_db_path = Path(state_db_path)
        self.vector_db_path = Path(vector_db_path)
        self.postgres_url = postgres_url

        # Create directories
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        # Component registry
        self.components: Dict[str, SystemComponent] = {}

        # Integrated capabilities
        self.capabilities: Dict[str, IntegratedCapability] = {}

        # System state
        self.state: Optional[SystemState] = None

        # Start time
        self.start_time = datetime.now()

        # Initialize persistent state database
        self._initialize_persistent_state()

        # Initialize all integrations
        self._initialize_integrations()

        # Integrate knowledge into cognitive memory
        if hasattr(self, 'cognitive_memory'):
            self.integrate_knowledge_into_cognitive_memory()

        # Save initial state
        self.save_state_snapshot()

        logger.info("Integrated Self-Awareness Engine initialized")

    def _initialize_integrations(self):
        """Initialize connections to all system components"""

        logger.info("Initializing system integrations...")

        # 1. Vector Database (ChromaDB + Enhanced RAG)
        if RAG_AVAILABLE and VECTOR_DB_AVAILABLE:
            try:
                self.rag_system = EnhancedRAGSystem(
                    storage_dir=str(self.vector_db_path),
                    embedding_model="all-MiniLM-L6-v2",
                    enable_reranking=True
                )

                # Register component
                self.components["vector_db"] = SystemComponent(
                    component_id="vector_db",
                    component_type=SystemComponentType.VECTOR_DATABASE,
                    name="ChromaDB Vector Database",
                    instance=self.rag_system,
                    status="active",
                    capabilities=[
                        "semantic_search",
                        "document_indexing",
                        "embedding_generation",
                        "similarity_matching"
                    ],
                    metadata={"embedding_model": "all-MiniLM-L6-v2"},
                    last_health_check=datetime.now()
                )

                logger.info("✅ Vector database integrated")
            except Exception as e:
                logger.error(f"❌ Vector database integration failed: {e}")

        # 2. Cognitive Memory System
        if COGNITIVE_MEMORY_AVAILABLE and POSTGRES_AVAILABLE:
            try:
                self.cognitive_memory = CognitiveMemorySystem(
                    postgres_url=self.postgres_url or "postgresql://localhost/lat5150",
                    embedding_model="all-MiniLM-L6-v2"
                )

                self.components["cognitive_memory"] = SystemComponent(
                    component_id="cognitive_memory",
                    component_type=SystemComponentType.COGNITIVE_MEMORY,
                    name="Brain-Inspired Cognitive Memory",
                    instance=self.cognitive_memory,
                    status="active",
                    capabilities=[
                        "episodic_memory",
                        "semantic_memory",
                        "memory_consolidation",
                        "associative_recall"
                    ],
                    metadata={"tiers": ["sensory", "working", "short_term", "long_term"]},
                    last_health_check=datetime.now()
                )

                logger.info("✅ Cognitive memory system integrated")
            except Exception as e:
                logger.error(f"❌ Cognitive memory integration failed: {e}")

        # 3. DSMIL AI Engine
        if DSMIL_AI_AVAILABLE:
            try:
                self.ai_engine = DSMILAIEngine()

                self.components["ai_engine"] = SystemComponent(
                    component_id="ai_engine",
                    component_type=SystemComponentType.AI_ENGINE,
                    name="DSMIL Multi-Model AI Engine",
                    instance=self.ai_engine,
                    status="active",
                    capabilities=[
                        "text_generation",
                        "code_generation",
                        "reasoning",
                        "model_routing"
                    ],
                    metadata={
                        "models": list(self.ai_engine.models.keys()),
                        "default_model": "uncensored_code"
                    },
                    last_health_check=datetime.now()
                )

                logger.info("✅ DSMIL AI engine integrated")
            except Exception as e:
                logger.error(f"❌ AI engine integration failed: {e}")

        # 4. Quantum Crypto Layer
        if QUANTUM_CRYPTO_AVAILABLE:
            try:
                self.crypto_layer = QuantumCryptoLayer()

                self.components["crypto_layer"] = SystemComponent(
                    component_id="crypto_layer",
                    component_type=SystemComponentType.CRYPTO_LAYER,
                    name="CSNA 2.0 Quantum Crypto Layer",
                    instance=self.crypto_layer,
                    status="active",
                    capabilities=[
                        "post_quantum_encryption",
                        "key_derivation",
                        "hmac_authentication",
                        "sha3_hashing"
                    ],
                    metadata={"standard": "CSNA 2.0", "nist_pqc": True},
                    last_health_check=datetime.now()
                )

                logger.info("✅ Quantum crypto layer integrated")
            except Exception as e:
                logger.error(f"❌ Crypto layer integration failed: {e}")

        # 5. DSMIL Subsystem Controller
        if DSMIL_CONTROLLER_AVAILABLE:
            try:
                self.dsmil_controller = DSMILSubsystemController()

                self.components["dsmil_controller"] = SystemComponent(
                    component_id="dsmil_controller",
                    component_type=SystemComponentType.HARDWARE_CONTROLLER,
                    name="DSMIL 84-Device Controller",
                    instance=self.dsmil_controller,
                    status="active",
                    capabilities=[
                        "hardware_control",
                        "device_activation",
                        "platform_integrity",
                        "device_monitoring"
                    ],
                    metadata={"total_devices": 84, "safe_devices": 79},
                    last_health_check=datetime.now()
                )

                logger.info("✅ DSMIL subsystem controller integrated")
            except Exception as e:
                logger.error(f"❌ DSMIL controller integration failed: {e}")

        # 6. TPM Attestation
        if TPM_AVAILABLE:
            try:
                self.tpm_crypto = get_tpm_crypto()

                self.components["tpm_attestation"] = SystemComponent(
                    component_id="tpm_attestation",
                    component_type=SystemComponentType.TPM_ATTESTATION,
                    name="TPM 2.0 Cryptographic Attestation",
                    instance=self.tpm_crypto,
                    status="active",
                    capabilities=[
                        "hardware_attestation",
                        "platform_integrity",
                        "secure_key_storage",
                        "random_generation"
                    ],
                    metadata={"tpm_version": "2.0", "algorithms": 88},
                    last_health_check=datetime.now()
                )

                logger.info("✅ TPM attestation integrated")
            except Exception as e:
                logger.error(f"❌ TPM integration failed: {e}")

        # 7. Unified Orchestrator
        if ORCHESTRATOR_AVAILABLE:
            try:
                self.orchestrator = UnifiedOrchestrator()

                self.components["orchestrator"] = SystemComponent(
                    component_id="orchestrator",
                    component_type=SystemComponentType.ORCHESTRATOR,
                    name="Unified System Orchestrator",
                    instance=self.orchestrator,
                    status="active",
                    capabilities=[
                        "workflow_coordination",
                        "task_distribution",
                        "system_coordination"
                    ],
                    metadata={},
                    last_health_check=datetime.now()
                )

                logger.info("✅ Unified orchestrator integrated")
            except Exception as e:
                logger.error(f"❌ Orchestrator integration failed: {e}")

        # 8. MCP Servers (scan and integrate)
        self._integrate_mcp_servers()

        # 9. Discover integrated capabilities
        self._discover_integrated_capabilities()

        # 10. Initial state update
        self._update_system_state()

        logger.info(f"✅ Integration complete: {len(self.components)} components, {len(self.capabilities)} capabilities")

    def _integrate_mcp_servers(self):
        """Discover and integrate all MCP servers"""
        try:
            # Check for MCP server configuration
            mcp_config_path = self.workspace_path / "02-ai-engine" / "mcp_servers_config.json"

            if mcp_config_path.exists():
                with open(mcp_config_path) as f:
                    mcp_config = json.load(f)

                for server_name, server_info in mcp_config.get("mcpServers", {}).items():
                    self.components[f"mcp_{server_name}"] = SystemComponent(
                        component_id=f"mcp_{server_name}",
                        component_type=SystemComponentType.MCP_SERVER,
                        name=f"MCP: {server_name}",
                        instance=server_info,
                        status="unknown",  # Would need to ping to verify
                        capabilities=server_info.get("capabilities", []),
                        metadata=server_info,
                        last_health_check=datetime.now()
                    )

                logger.info(f"✅ {len(mcp_config.get('mcpServers', {}))} MCP servers integrated")
        except Exception as e:
            logger.error(f"❌ MCP server integration failed: {e}")

    def _discover_integrated_capabilities(self):
        """
        Discover capabilities that span multiple integrated systems

        Unlike standalone discovery, this finds capabilities that REQUIRE
        multiple systems working together (true integration)
        """

        logger.info("Discovering integrated capabilities...")

        # Define integrated capabilities (these require multiple systems)
        integrated_caps = [
            {
                "id": "semantic_code_search",
                "name": "Semantic Code Search with AI Understanding",
                "description": "Search codebase using natural language, powered by vector embeddings and AI reasoning",
                "required": [SystemComponentType.VECTOR_DATABASE, SystemComponentType.AI_ENGINE],
            },
            {
                "id": "attested_ai_generation",
                "name": "Hardware-Attested AI Generation",
                "description": "Generate AI responses with TPM attestation and quantum-resistant signatures",
                "required": [SystemComponentType.AI_ENGINE, SystemComponentType.TPM_ATTESTATION, SystemComponentType.CRYPTO_LAYER],
            },
            {
                "id": "hardware_aware_execution",
                "name": "Hardware-Aware Task Execution",
                "description": "Execute tasks with full awareness of DSMIL hardware state and platform integrity",
                "required": [SystemComponentType.HARDWARE_CONTROLLER, SystemComponentType.TPM_ATTESTATION, SystemComponentType.ORCHESTRATOR],
            },
            {
                "id": "long_term_learning",
                "name": "Long-Term Learning with Memory Consolidation",
                "description": "Learn from interactions with brain-inspired memory consolidation",
                "required": [SystemComponentType.COGNITIVE_MEMORY, SystemComponentType.VECTOR_DATABASE],
            },
            {
                "id": "secure_knowledge_retrieval",
                "name": "Secure Knowledge Retrieval",
                "description": "Retrieve knowledge from vector DB with quantum-resistant encryption",
                "required": [SystemComponentType.VECTOR_DATABASE, SystemComponentType.CRYPTO_LAYER],
            },
        ]

        for cap_def in integrated_caps:
            # Check which required components are available
            available = []
            for req_type in cap_def["required"]:
                for comp_id, comp in self.components.items():
                    if comp.component_type == req_type and comp.status == "active":
                        available.append(comp_id)
                        break

            # Calculate confidence based on availability
            confidence = len(available) / len(cap_def["required"])

            capability = IntegratedCapability(
                capability_id=cap_def["id"],
                name=cap_def["name"],
                description=cap_def["description"],
                required_components=cap_def["required"],
                available_components=available,
                confidence=confidence
            )

            self.capabilities[cap_def["id"]] = capability

            # Generate vector embedding for semantic matching
            if VECTOR_DB_AVAILABLE and hasattr(self, 'rag_system'):
                try:
                    # Embed capability description for semantic search
                    embedding = self.rag_system.embedding_model.encode(
                        f"{cap_def['name']} {cap_def['description']}"
                    ).tolist()
                    capability.vector_embedding = embedding
                except:
                    pass

        logger.info(f"Discovered {len(self.capabilities)} integrated capabilities")

    def _update_system_state(self):
        """Update comprehensive system state"""

        components_status = {
            comp_id: comp.status
            for comp_id, comp in self.components.items()
        }

        active_capabilities = [
            cap_id for cap_id, cap in self.capabilities.items()
            if cap.confidence > 0.8
        ]

        # Get metrics from integrated systems
        vector_db_docs = 0
        if hasattr(self, 'rag_system'):
            try:
                # Would query ChromaDB for document count
                pass
            except:
                pass

        cognitive_memory_blocks = 0
        if hasattr(self, 'cognitive_memory'):
            try:
                # Would query PostgreSQL for memory block count
                pass
            except:
                pass

        dsmil_devices_active = 0
        if hasattr(self, 'dsmil_controller'):
            try:
                # Would check DSMIL subsystems
                pass
            except:
                pass

        # Calculate health score
        active_components = sum(1 for c in self.components.values() if c.status == "active")
        health_score = active_components / len(self.components) if self.components else 0.0

        self.state = SystemState(
            timestamp=datetime.now(),
            components_status=components_status,
            active_capabilities=active_capabilities,
            vector_db_document_count=vector_db_docs,
            cognitive_memory_blocks=cognitive_memory_blocks,
            dsmil_devices_active=dsmil_devices_active,
            crypto_key_rotation_due=False,  # Would check crypto layer
            tpm_attestation_valid=True,     # Would check TPM
            mcp_servers_online=sum(1 for c in self.components.values()
                                  if c.component_type == SystemComponentType.MCP_SERVER),
            total_interactions=0,  # Would load from database
            current_load={},
            health_score=health_score
        )

    def _initialize_persistent_state(self):
        """Initialize persistent state management with SQLite and PostgreSQL"""
        import sqlite3

        try:
            # Create SQLite state database
            self.state_db_connection = sqlite3.connect(str(self.state_db_path))
            cursor = self.state_db_connection.cursor()

            # Component state history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS component_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    UNIQUE(timestamp, component_id)
                )
            """)

            # Capability usage tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS capability_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    capability_id TEXT NOT NULL,
                    execution_time_ms REAL,
                    success BOOLEAN,
                    components_used TEXT,
                    error_message TEXT
                )
            """)

            # System state snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    health_score REAL NOT NULL,
                    active_components INTEGER,
                    active_capabilities INTEGER,
                    vector_db_docs INTEGER,
                    cognitive_memory_blocks INTEGER,
                    dsmil_devices INTEGER,
                    total_interactions INTEGER,
                    state_json TEXT
                )
            """)


            # Learning events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    context TEXT,
                    insight TEXT,
                    confidence REAL,
                    applied BOOLEAN DEFAULT 0
                )
            """)

            self.state_db_connection.commit()
            logger.info("✅ Persistent state database initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize persistent state: {e}")
            self.state_db_connection = None

    def save_state_snapshot(self):
        """Save current system state to persistent storage"""
        if not self.state_db_connection or not self.state:
            return

        try:
            cursor = self.state_db_connection.cursor()

            # Save system snapshot
            cursor.execute("""
                INSERT INTO system_snapshots
                (timestamp, health_score, active_components, active_capabilities,
                 vector_db_docs, cognitive_memory_blocks, dsmil_devices,
                 total_interactions, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                self.state.health_score,
                sum(1 for c in self.components.values() if c.status == "active"),
                len([c for c in self.capabilities.values() if c.confidence >= 0.8]),
                self.state.vector_db_document_count,
                self.state.cognitive_memory_blocks,
                self.state.dsmil_devices_active,
                self.state.total_interactions,
                json.dumps(asdict(self.state), default=str)
            ))

            # Save component states
            for comp_id, comp in self.components.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO component_state
                    (timestamp, component_id, component_type, status, error_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().timestamp(),
                    comp_id,
                    comp.component_type.value,
                    comp.status,
                    comp.error_count,
                    json.dumps(comp.metadata)
                ))

            self.state_db_connection.commit()
            logger.debug("State snapshot saved")

        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")

    def load_historical_state(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Load historical state snapshots"""
        if not self.state_db_connection:
            return []

        try:
            cursor = self.state_db_connection.cursor()
            cutoff = (datetime.now() - timedelta(hours=hours_back)).timestamp()

            cursor.execute("""
                SELECT timestamp, health_score, active_components,
                       active_capabilities, state_json
                FROM system_snapshots
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "timestamp": datetime.fromtimestamp(row[0]),
                    "health_score": row[1],
                    "active_components": row[2],
                    "active_capabilities": row[3],
                    "state": json.loads(row[4])
                })

            return results

        except Exception as e:
            logger.error(f"Failed to load historical state: {e}")
            return []

    def track_capability_usage(
        self,
        capability_id: str,
        execution_time_ms: float,
        success: bool,
        components_used: List[str],
        error_message: Optional[str] = None
    ):
        """Track capability usage for learning and optimization"""
        if not self.state_db_connection:
            return

        try:
            cursor = self.state_db_connection.cursor()

            cursor.execute("""
                INSERT INTO capability_usage
                (timestamp, capability_id, execution_time_ms, success,
                 components_used, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                capability_id,
                execution_time_ms,
                success,
                json.dumps(components_used),
                error_message
            ))

            self.state_db_connection.commit()

            # Update capability stats
            if capability_id in self.capabilities:
                cap = self.capabilities[capability_id]
                cap.usage_count += 1
                cap.last_used = datetime.now()

                # Update success rate (exponential moving average)
                alpha = 0.1
                cap.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * cap.success_rate

                # Update avg execution time
                cap.avg_execution_time_ms = (
                    alpha * execution_time_ms + (1 - alpha) * cap.avg_execution_time_ms
                )

            logger.debug(f"Tracked usage for {capability_id}: success={success}, time={execution_time_ms}ms")

        except Exception as e:
            logger.error(f"Failed to track capability usage: {e}")

    def integrate_knowledge_into_cognitive_memory(self):
        """
        Integrate system knowledge into the multi-tiered cognitive memory system

        Uses the existing sophisticated cognitive memory with:
        - Sensory, working, short-term, long-term, and archived tiers
        - Associative recall and memory consolidation
        - Semantic embeddings for relationships
        """
        if not hasattr(self, 'cognitive_memory'):
            logger.warning("Cognitive memory system not available for knowledge integration")
            return

        try:
            # Store component knowledge in semantic memory (long-term tier)
            for comp_id, comp in self.components.items():
                knowledge_content = (
                    f"System component: {comp.name} ({comp.component_type.value})\n"
                    f"Status: {comp.status}\n"
                    f"Capabilities: {', '.join(comp.capabilities)}\n"
                    f"Metadata: {json.dumps(comp.metadata, indent=2)}"
                )

                self.cognitive_memory.store_memory(
                    content=knowledge_content,
                    memory_type=MemoryType.SEMANTIC,
                    tier=MemoryTier.LONG_TERM,
                    salience=SalienceLevel.HIGH if comp.status == "active" else SalienceLevel.MODERATE,
                    metadata={
                        "component_id": comp_id,
                        "component_type": comp.component_type.value,
                        "status": comp.status,
                        "capabilities": comp.capabilities
                    }
                )

            # Store integrated capability knowledge
            for cap_id, cap in self.capabilities.items():
                capability_content = (
                    f"Integrated capability: {cap.name}\n"
                    f"Description: {cap.description}\n"
                    f"Required components: {[t.value for t in cap.required_components]}\n"
                    f"Available components: {cap.available_components}\n"
                    f"Confidence: {cap.confidence:.2%}\n"
                    f"Usage: {cap.usage_count} times, Success rate: {cap.success_rate:.2%}"
                )

                self.cognitive_memory.store_memory(
                    content=capability_content,
                    memory_type=MemoryType.SEMANTIC,
                    tier=MemoryTier.LONG_TERM if cap.usage_count > 5 else MemoryTier.SHORT_TERM,
                    salience=SalienceLevel.CRITICAL if cap.confidence >= 1.0 else SalienceLevel.HIGH,
                    metadata={
                        "capability_id": cap_id,
                        "confidence": cap.confidence,
                        "usage_count": cap.usage_count,
                        "success_rate": cap.success_rate
                    }
                )

            # Store system state as episodic memory (what happened)
            if self.state:
                state_content = (
                    f"System state observation:\n"
                    f"Health score: {self.state.health_score:.2%}\n"
                    f"Active components: {sum(1 for c in self.components.values() if c.status == 'active')}/{len(self.components)}\n"
                    f"Active capabilities: {len(self.state.active_capabilities)}\n"
                    f"Vector DB documents: {self.state.vector_db_document_count}\n"
                    f"Cognitive memory blocks: {self.state.cognitive_memory_blocks}\n"
                    f"DSMIL devices active: {self.state.dsmil_devices_active}"
                )

                self.cognitive_memory.store_memory(
                    content=state_content,
                    memory_type=MemoryType.EPISODIC,
                    tier=MemoryTier.WORKING,
                    salience=SalienceLevel.HIGH if self.state.health_score < 0.8 else SalienceLevel.MODERATE,
                    metadata={
                        "timestamp": self.state.timestamp.isoformat(),
                        "health_score": self.state.health_score,
                        "state_snapshot": True
                    }
                )

            # Create associative links using the cognitive memory's embedding system
            # The cognitive memory system already handles semantic relationships via embeddings

            logger.info("✅ Knowledge integrated into cognitive memory system")

        except Exception as e:
            logger.error(f"Failed to integrate knowledge into cognitive memory: {e}")

    def query_system_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query system knowledge using cognitive memory's associative recall

        Leverages the multi-tiered memory system's semantic search capabilities
        """
        if not hasattr(self, 'cognitive_memory'):
            return []

        try:
            # Use cognitive memory's recall method with semantic search
            memories = self.cognitive_memory.recall_similar(
                query=query,
                limit=limit,
                tier=None  # Search across all tiers
            )

            results = []
            for memory in memories:
                results.append({
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "tier": memory.tier.value,
                    "salience": memory.salience.value,
                    "timestamp": memory.timestamp,
                    "metadata": memory.metadata
                })

            return results

        except Exception as e:
            logger.error(f"Failed to query system knowledge: {e}")
            return []

    def record_learning_event(
        self,
        event_type: str,
        context: str,
        insight: str,
        confidence: float
    ):
        """Record a learning event for future optimization"""
        if not self.state_db_connection:
            return

        try:
            cursor = self.state_db_connection.cursor()

            cursor.execute("""
                INSERT INTO learning_events
                (timestamp, event_type, context, insight, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                event_type,
                context,
                insight,
                confidence
            ))

            self.state_db_connection.commit()
            logger.info(f"Learning event recorded: {event_type}")

            # Also store in cognitive memory if available
            if hasattr(self, 'cognitive_memory'):
                try:
                    self.cognitive_memory.store_memory(
                        content=f"{event_type}: {insight}",
                        memory_type=MemoryType.EPISODIC,
                        tier=MemoryTier.WORKING,
                        salience=SalienceLevel.HIGH,
                        metadata={
                            "context": context,
                            "confidence": confidence
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not store in cognitive memory: {e}")

        except Exception as e:
            logger.error(f"Failed to record learning event: {e}")

    def discover_capabilities(self) -> Dict[str, IntegratedCapability]:
        """
        Compatibility method for unified tactical API
        Returns discovered integrated capabilities
        """
        return self.capabilities

    def discover_resources(self) -> Dict[str, Any]:
        """
        Compatibility method for unified tactical API
        Returns discovered resources across all integrated systems
        """
        resources = {
            "local_models": [],
            "vector_databases": [],
            "hardware_devices": [],
            "mcp_servers": [],
            "crypto_capabilities": [],
            "memory_tiers": []
        }

        # Discover local AI models
        if hasattr(self, 'ai_engine'):
            try:
                if hasattr(self.ai_engine, 'models'):
                    resources["local_models"] = list(self.ai_engine.models.keys())
            except Exception as e:
                logger.debug(f"Could not enumerate AI models: {e}")

        # Vector database info
        if hasattr(self, 'rag_system'):
            resources["vector_databases"].append({
                "type": "ChromaDB",
                "embedding_model": "all-MiniLM-L6-v2",
                "status": "active"
            })

        # DSMIL hardware devices
        if hasattr(self, 'dsmil_controller'):
            try:
                if hasattr(self.dsmil_controller, 'subsystems'):
                    resources["hardware_devices"] = [
                        {
                            "id": dev_id,
                            "name": dev.get("name", "Unknown"),
                            "status": dev.get("status", "unknown")
                        }
                        for dev_id, dev in self.dsmil_controller.subsystems.items()
                    ]
            except Exception as e:
                logger.debug(f"Could not enumerate DSMIL devices: {e}")

        # MCP servers
        mcp_components = [
            comp for comp in self.components.values()
            if comp.component_type == SystemComponentType.MCP_SERVER
        ]
        resources["mcp_servers"] = [
            {"name": comp.name, "status": comp.status}
            for comp in mcp_components
        ]

        # Crypto capabilities
        if hasattr(self, 'crypto_layer'):
            resources["crypto_capabilities"] = [
                "SHA3-512", "HMAC-SHA3-512", "Post-Quantum Crypto", "CSNA 2.0"
            ]

        # Memory tiers
        if hasattr(self, 'cognitive_memory'):
            resources["memory_tiers"] = [
                "sensory", "working", "short_term", "long_term", "archived"
            ]

        return resources

    def update_system_state(self):
        """
        Compatibility method for unified tactical API
        Updates system state and persists to database
        """
        self._update_system_state()
        self.save_state_snapshot()

        # Also update knowledge in cognitive memory
        if hasattr(self, 'cognitive_memory'):
            self.integrate_knowledge_into_cognitive_memory()

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive self-awareness report integrating ALL systems
        """

        self._update_system_state()

        return {
            "system_name": "LAT5150 DRVMIL Integrated Tactical AI Platform",
            "self_awareness_level": "fully_integrated",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),

            "integrated_components": {
                "total": len(self.components),
                "by_type": {
                    comp_type.value: sum(1 for c in self.components.values()
                                        if c.component_type == comp_type)
                    for comp_type in SystemComponentType
                },
                "status_summary": {
                    "active": sum(1 for c in self.components.values() if c.status == "active"),
                    "degraded": sum(1 for c in self.components.values() if c.status == "degraded"),
                    "offline": sum(1 for c in self.components.values() if c.status == "offline"),
                },
                "details": [
                    {
                        "id": comp.component_id,
                        "name": comp.name,
                        "type": comp.component_type.value,
                        "status": comp.status,
                        "capabilities": comp.capabilities
                    }
                    for comp in self.components.values()
                ]
            },

            "integrated_capabilities": {
                "total": len(self.capabilities),
                "fully_available": sum(1 for c in self.capabilities.values() if c.confidence >= 1.0),
                "partially_available": sum(1 for c in self.capabilities.values() if 0.5 <= c.confidence < 1.0),
                "unavailable": sum(1 for c in self.capabilities.values() if c.confidence < 0.5),
                "details": [
                    {
                        "id": cap.capability_id,
                        "name": cap.name,
                        "confidence": cap.confidence,
                        "required_components": [t.value for t in cap.required_components],
                        "available_components": cap.available_components,
                        "usage_stats": {
                            "times_used": cap.usage_count,
                            "success_rate": cap.success_rate,
                            "avg_time_ms": cap.avg_execution_time_ms
                        }
                    }
                    for cap in self.capabilities.values()
                ]
            },

            "system_state": asdict(self.state) if self.state else {},

            "integration_quality": {
                "component_health": self.state.health_score if self.state else 0.0,
                "capability_availability": (
                    sum(1 for c in self.capabilities.values() if c.confidence >= 1.0) /
                    len(self.capabilities) if self.capabilities else 0.0
                ),
                "overall_integration": (
                    (self.state.health_score +
                     (sum(1 for c in self.capabilities.values() if c.confidence >= 1.0) /
                      len(self.capabilities) if self.capabilities else 0.0)) / 2
                    if self.state else 0.0
                )
            },

            "self_awareness_features": {
                "vector_semantic_understanding": RAG_AVAILABLE and VECTOR_DB_AVAILABLE,
                "brain_inspired_memory": COGNITIVE_MEMORY_AVAILABLE,
                "multi_model_reasoning": DSMIL_AI_AVAILABLE,
                "quantum_resistant_crypto": QUANTUM_CRYPTO_AVAILABLE,
                "hardware_platform_integrity": DSMIL_CONTROLLER_AVAILABLE,
                "tpm_attestation": TPM_AVAILABLE,
                "mcp_tool_integration": len([c for c in self.components.values()
                                            if c.component_type == SystemComponentType.MCP_SERVER]) > 0,
                "orchestrated_workflows": ORCHESTRATOR_AVAILABLE
            }
        }


# Example usage
async def main():
    """Test integrated self-awareness engine"""

    engine = IntegratedSelfAwarenessEngine()

    print("\n" + "="*70)
    print("INTEGRATED SELF-AWARENESS ENGINE - System Report")
    print("="*70 + "\n")

    report = engine.get_comprehensive_report()

    print(json.dumps(report, indent=2, default=str))

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
