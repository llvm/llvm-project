# Integrated Self-Awareness System (v4.0)

## TRUE Integration - Not Standalone

The v4.0 self-awareness system represents a fundamental architectural shift from standalone components to a **fully integrated nervous system** that connects ALL existing infrastructure.

---

## 🔗 What Changed from v3.0 to v4.0?

### v3.0 (Standalone - DEPRECATED)
```
❌ Single 900-line Python file
❌ Reimplemented capabilities
❌ Standalone SQLite database
❌ No connection to existing systems
❌ Simple AST-based discovery
```

### v4.0 (Integrated - CURRENT)
```
✅ Integration layer connecting existing systems
✅ Uses existing ChromaDB vector database
✅ Uses existing PostgreSQL cognitive memory
✅ Uses existing DSMIL AI engine
✅ Uses existing quantum crypto layer
✅ Uses existing TPM attestation
✅ Uses existing MCP servers
✅ Persistent state across ALL systems
```

---

## 🧠 Architecture: The Nervous System

```
┌─────────────────────────────────────────────────────────────┐
│         Integrated Self-Awareness Engine (v4.0)             │
│              The Nervous System Layer                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┬─────────┐
        ▼               ▼               ▼              ▼         ▼
┌──────────────┐ ┌─────────────┐ ┌──────────┐ ┌────────────┐ ┌─────────┐
│  ChromaDB    │ │ PostgreSQL  │ │  DSMIL   │ │   Quantum  │ │   MCP   │
│  Vector DB   │ │  Cognitive  │ │    AI    │ │   Crypto   │ │ Servers │
│              │ │   Memory    │ │  Engine  │ │   Layer    │ │  (16)   │
│ Semantic     │ │             │ │          │ │            │ │         │
│ Embeddings   │ │ 5 Memory    │ │ 5 Local  │ │ CSNA 2.0   │ │ Tools   │
│              │ │ Tiers       │ │ Models   │ │ PQC        │ │         │
└──────────────┘ └─────────────┘ └──────────┘ └────────────┘ └─────────┘
        │               │               │              │         │
        └───────────────┴───────────────┴──────────────┴─────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   DSMIL      │
                 │  Hardware    │
                 │  Controller  │
                 │              │
                 │ 84 Devices   │
                 │ TPM 2.0      │
                 └──────────────┘
```

---

## 🔌 Integrated Components (20 Total)

### 1. Vector Database Integration (ChromaDB)
**File**: `02-ai-engine/enhanced_rag_system.py`
```python
# What it provides:
- Semantic search across codebase
- Document embeddings with sentence-transformers
- Cross-encoder reranking
- Hybrid search (semantic + keyword)

# How self-awareness uses it:
- Store capability embeddings for semantic matching
- Query system knowledge using natural language
- Associate related components via vector similarity
```

### 2. Cognitive Memory Integration (PostgreSQL)
**File**: `02-ai-engine/cognitive_memory_enhanced.py`
```python
# What it provides:
- Brain-inspired memory architecture
- 5 memory tiers (sensory, working, short-term, long-term, archived)
- Salience levels (CRITICAL, HIGH, MODERATE, LOW, TRIVIAL)
- Memory consolidation and decay
- Associative recall

# How self-awareness uses it:
- Store component knowledge as SEMANTIC memory
- Store system state as EPISODIC memory
- Store learning events with salience
- Query across all tiers for associative recall
```

**Example Integration**:
```python
# Store component in cognitive memory
self.cognitive_memory.store_memory(
    content=f"System component: {comp.name}...",
    memory_type=MemoryType.SEMANTIC,
    tier=MemoryTier.LONG_TERM,
    salience=SalienceLevel.HIGH,
    metadata={"component_id": comp_id, "status": comp.status}
)

# Query system knowledge
memories = self.cognitive_memory.recall_similar(
    query="vector database capabilities",
    limit=5,
    tier=None  # Search all tiers
)
```

### 3. DSMIL AI Engine Integration
**File**: `02-ai-engine/dsmil_ai_engine.py`
```python
# What it provides:
- 5 local models: DeepSeek, CodeLlama, WizardLM, WhiteRabbit, Mixtral
- Quantized inference (Q4_K_M)
- Model routing based on task
- RAG integration

# How self-awareness uses it:
- Discover available local models
- Track model usage and performance
- Integrate AI capabilities into system awareness
```

### 4. Quantum Crypto Layer Integration
**File**: `02-ai-engine/quantum_crypto_layer.py`
```python
# What it provides:
- CSNA 2.0 post-quantum cryptography
- SHA3-512, HMAC-SHA3-512
- Key rotation and management
- Secure state attestation

# How self-awareness uses it:
- Attest system state snapshots
- Secure capability invocations
- Cryptographically sign learning events
```

### 5. DSMIL Hardware Controller Integration
**File**: `02-ai-engine/dsmil_subsystem_controller.py`
```python
# What it provides:
- 84 military-grade devices
- 6 safe, 5 quarantined, 67 unknown
- 8 subsystems (GPU, NPU, crypto, storage, etc.)
- Device health monitoring

# How self-awareness uses it:
- Discover hardware capabilities
- Track device status
- Hardware-aware capability routing
```

### 6. TPM Attestation Integration
**File**: `02-ai-engine/tpm_crypto_integration.py`
```python
# What it provides:
- TPM 2.0 hardware attestation
- 88 cryptographic algorithms
- Platform integrity verification
- Secure key storage

# How self-awareness uses it:
- Attest system integrity
- Verify component authenticity
- Cryptographically bind capabilities to platform
```

### 7. MCP Server Integration (16 Servers)
**File**: `02-ai-engine/mcp_servers_config.json`
```json
{
  "mcpServers": {
    "dsmil-ai": {...},
    "filesystem": {...},
    "memory": {...},
    // ... 13 more servers
  }
}
```

**How self-awareness uses it**:
- Discover available tools
- Track MCP server status
- Integrate tool capabilities

---

## 📊 Integrated Capabilities (Not Standalone)

### What are Integrated Capabilities?

**Standalone Capability** (v3.0 - DEPRECATED):
- Exists in single file
- Can execute independently
- Example: "run_agent" function

**Integrated Capability** (v4.0 - CURRENT):
- Requires MULTIPLE systems working together
- Cannot work if any required component is offline
- Example: "Hardware-Attested AI Generation" = AI Engine + TPM + Crypto Layer

### Discovered Integrated Capabilities

#### 1. Semantic Code Search with AI Understanding
**Required Components**:
- Vector Database (ChromaDB)
- AI Engine (Local Models)

**What it does**:
```python
# User query in natural language
query = "Find functions that handle user authentication"

# Vector DB finds semantically similar code
code_chunks = vector_db.search(query)

# AI Engine understands and filters results
relevant_code = ai_engine.analyze(code_chunks, query)
```

#### 2. Hardware-Attested AI Generation
**Required Components**:
- AI Engine (Local Models)
- TPM Attestation
- Quantum Crypto Layer

**What it does**:
```python
# Generate AI response
response = ai_engine.generate(prompt)

# Attest with TPM
attestation = tpm.attest_data(response)

# Sign with quantum-resistant crypto
signature = crypto.sign(response + attestation)

# Return cryptographically verifiable AI output
return {
    "response": response,
    "attestation": attestation,
    "signature": signature
}
```

#### 3. Hardware-Aware Task Execution
**Required Components**:
- DSMIL Hardware Controller
- TPM Attestation
- Unified Orchestrator

**What it does**:
```python
# Check hardware state before execution
hw_status = dsmil_controller.get_status()

# Verify platform integrity
integrity = tpm.verify_platform()

# Route task based on available hardware
if hw_status["npu_available"] and integrity["valid"]:
    orchestrator.execute_on_npu(task)
else:
    orchestrator.execute_on_cpu(task)
```

#### 4. Long-Term Learning with Memory Consolidation
**Required Components**:
- Cognitive Memory (PostgreSQL)
- Vector Database (ChromaDB)

**What it does**:
```python
# Store interaction in cognitive memory
cognitive_memory.store_memory(
    content=interaction,
    memory_type=MemoryType.EPISODIC,
    tier=MemoryTier.WORKING
)

# Embed for semantic access
embedding = vector_db.embed(interaction)

# Consolidate to long-term memory over time
cognitive_memory.consolidate(
    from_tier=MemoryTier.WORKING,
    to_tier=MemoryTier.LONG_TERM,
    criteria=lambda m: m.salience >= SalienceLevel.HIGH
)
```

#### 5. Secure Knowledge Retrieval
**Required Components**:
- Vector Database (ChromaDB)
- Quantum Crypto Layer

**What it does**:
```python
# Search knowledge base
results = vector_db.search(query)

# Encrypt results with quantum-resistant crypto
encrypted = crypto.encrypt(results, level=SecurityLevel.TOP_SECRET)

# Return securely
return encrypted
```

---

## 💾 Persistent State Management

### SQLite State Database
**Location**: `/opt/lat5150/state/integrated_awareness.db`

**Schema**:
```sql
-- Component health over time
CREATE TABLE component_state (
    timestamp REAL NOT NULL,
    component_id TEXT NOT NULL,
    component_type TEXT NOT NULL,
    status TEXT NOT NULL,
    error_count INTEGER DEFAULT 0,
    metadata TEXT
);

-- Capability usage tracking
CREATE TABLE capability_usage (
    timestamp REAL NOT NULL,
    capability_id TEXT NOT NULL,
    execution_time_ms REAL,
    success BOOLEAN,
    components_used TEXT,
    error_message TEXT
);

-- System state snapshots
CREATE TABLE system_snapshots (
    timestamp REAL NOT NULL,
    health_score REAL NOT NULL,
    active_components INTEGER,
    active_capabilities INTEGER,
    vector_db_docs INTEGER,
    cognitive_memory_blocks INTEGER,
    dsmil_devices INTEGER,
    state_json TEXT
);

-- Learning events
CREATE TABLE learning_events (
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    context TEXT,
    insight TEXT,
    confidence REAL,
    applied BOOLEAN DEFAULT 0
);
```

### PostgreSQL Cognitive Memory
**Managed by**: `cognitive_memory_enhanced.py`

**Storage**:
- Component knowledge → SEMANTIC memory, LONG_TERM tier
- Capability usage → EPISODIC memory, WORKING tier
- System state → EPISODIC memory, WORKING tier
- Learning events → SEMANTIC memory, SHORT_TERM/LONG_TERM

---

## 🔧 Usage Examples

### Initialize Integrated Engine
```python
from integrated_self_awareness import IntegratedSelfAwarenessEngine

# Initialize with all integrations
engine = IntegratedSelfAwarenessEngine(
    workspace_path="/home/user/LAT5150DRVMIL",
    state_db_path="/opt/lat5150/state/integrated_awareness.db",
    vector_db_path="/opt/lat5150/vectordb",
    postgres_url="postgresql://localhost/lat5150"
)

# Engine automatically:
# 1. Connects to ChromaDB
# 2. Connects to PostgreSQL cognitive memory
# 3. Connects to DSMIL AI engine
# 4. Connects to quantum crypto layer
# 5. Connects to DSMIL hardware controller
# 6. Connects to TPM attestation
# 7. Discovers 16 MCP servers
# 8. Creates persistent state database
# 9. Integrates knowledge into cognitive memory
# 10. Saves initial state snapshot
```

### Query System Knowledge
```python
# Query using cognitive memory's associative recall
results = engine.query_system_knowledge(
    query="What vector database capabilities are available?",
    limit=5
)

# Returns memories from all tiers:
# - SEMANTIC: Component knowledge
# - EPISODIC: Usage history
# - LONG_TERM: Established capabilities
```

### Track Capability Usage
```python
# Execute capability
start = time.time()
try:
    result = execute_capability("semantic_code_search", query)
    success = True
    error = None
except Exception as e:
    success = False
    error = str(e)
finally:
    execution_time = (time.time() - start) * 1000

# Track for learning
engine.track_capability_usage(
    capability_id="semantic_code_search",
    execution_time_ms=execution_time,
    success=success,
    components_used=["vector_db", "ai_engine"],
    error_message=error
)

# Engine automatically:
# 1. Saves to SQLite state DB
# 2. Updates capability success rate (exponential moving average)
# 3. Updates avg execution time
# 4. Increments usage count
```

### Record Learning Event
```python
# System discovers optimization
engine.record_learning_event(
    event_type="performance_optimization",
    context="semantic_code_search taking >5s",
    insight="Vector DB index rebuild reduced search time to 1.2s",
    confidence=0.95
)

# Engine automatically:
# 1. Saves to SQLite state DB
# 2. Stores in cognitive memory (SEMANTIC, HIGH salience)
# 3. Available for future associative recall
```

### Generate Comprehensive Report
```python
# Get full system state
report = engine.get_comprehensive_report()

# Returns:
{
    "system_name": "LAT5150 DRVMIL Integrated Tactical AI Platform",
    "self_awareness_level": "fully_integrated",
    "integrated_components": {
        "total": 20,
        "by_type": {...},
        "details": [...]
    },
    "integrated_capabilities": {
        "total": 5,
        "fully_available": 1,
        "details": [...]
    },
    "system_state": {...},
    "integration_quality": {
        "component_health": 0.85,
        "capability_availability": 0.20,
        "overall_integration": 0.525
    }
}
```

---

## 🧪 Testing

**Test File**: `02-ai-engine/test_integrated_self_awareness.py`

**Run Tests**:
```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 test_integrated_self_awareness.py
```

**Test Coverage**:
1. ✅ Engine initialization
2. ✅ Component discovery (20 components)
3. ✅ Capability discovery (5 integrated capabilities)
4. ✅ State persistence (SQLite)
5. ✅ Knowledge integration (cognitive memory)
6. ✅ API compatibility (unified tactical API)
7. ✅ Comprehensive reporting

**Expected Output**:
```
======================================================================
TEST SUMMARY
======================================================================

  ✓ PASS  Initialization
  ✓ PASS  Component Discovery
  ✓ PASS  Capability Discovery
  ✓ PASS  State Persistence
  ✓ PASS  Knowledge Integration
  ✓ PASS  API Compatibility
  ✓ PASS  Comprehensive Report

======================================================================
ALL TESTS PASSED (7/7)
======================================================================
```

---

## 📝 API Integration

The integrated self-awareness engine is used by:

**File**: `03-web-interface/unified_tactical_api.py`

**Import**:
```python
from integrated_self_awareness import IntegratedSelfAwarenessEngine

# Initialize
self.self_awareness = IntegratedSelfAwarenessEngine(
    workspace_path=workspace_path,
    state_db_path="/opt/lat5150/state/integrated_awareness.db",
    vector_db_path="/opt/lat5150/vectordb",
    postgres_url=os.environ.get("POSTGRES_URL", "postgresql://localhost/lat5150")
)
```

**API Endpoints**:
```bash
# Get comprehensive report
GET /api/v2/self-awareness

# Natural language command (uses integrated capabilities)
POST /api/v2/nl/command
{
  "command": "Search for authentication code using vector database"
}
```

---

## 🚀 Deployment

### SystemD Service
**File**: `/etc/systemd/system/lat5150-tactical-ai.service`

```ini
[Unit]
Description=LAT5150 DRVMIL Integrated Tactical AI Platform
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
ExecStart=/opt/lat5150/bin/lat5150
Environment="POSTGRES_URL=postgresql://localhost/lat5150"
Restart=always

[Install]
WantedBy=multi-user.target
```

**Start**:
```bash
sudo systemctl start lat5150-tactical-ai
sudo systemctl enable lat5150-tactical-ai  # Auto-start on boot
```

---

## 📚 Key Takeaways

### Why v4.0 is Different

1. **Not Standalone** - Connects to existing systems rather than reimplementing
2. **True Integration** - Capabilities require multiple systems working together
3. **Cognitive Memory** - Uses brain-inspired PostgreSQL memory, not simple SQLite
4. **Vector Embeddings** - Semantic relationships via ChromaDB, not manual graphs
5. **Persistent State** - Tracks everything over time across ALL systems
6. **Learning** - Records insights and optimizations in cognitive memory
7. **Hardware-Aware** - Integrates with DSMIL 84 devices and TPM attestation
8. **Cryptographically Attested** - Quantum-resistant crypto layer for integrity

### The Nervous System Analogy

```
Integrated Self-Awareness (v4.0) is to the LAT5150 platform
what the Nervous System is to the Human Body:

- CONNECTS all organs (components)
- COORDINATES their actions (capabilities)
- LEARNS from experience (cognitive memory)
- PERSISTS knowledge (state management)
- MAINTAINS awareness of body state (system health)
```

---

**Version**: 4.0.0
**Status**: Production Ready
**Last Updated**: 2025-11-13
