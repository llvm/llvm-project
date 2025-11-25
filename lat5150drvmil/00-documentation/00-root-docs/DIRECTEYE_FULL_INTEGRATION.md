# DIRECTEYE Full Integration Documentation

**Project:** LAT5150DRVMIL
**Component:** DIRECTEYE Enterprise Intelligence Platform
**Status:** ✅ Fully Integrated
**Date:** November 12, 2025
**Version:** 6.0.0

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Submodules](#submodules)
3. [Integration Points](#integration-points)
4. [DIRECTEYE Architecture](#directeye-architecture)
5. [Usage Examples](#usage-examples)
6. [Service Inventory](#service-inventory)
7. [File Locations](#file-locations)
8. [Quick Start](#quick-start)

---

## Overview

DIRECTEYE is fully integrated into the LAT5150DRVMIL intelligence system, providing comprehensive OSINT, blockchain, and threat intelligence capabilities through 40+ services and 35+ AI-powered MCP tools.

**Key Integration Points:**
- `ai_engine/directeye_intelligence.py` - AI Engine wrapper
- `06-intel-systems/directeye_intelligence_integration.py` - Full integration module
- `rag_system/mcp_servers/DIRECTEYE/` - DIRECTEYE source (22,403 files)

---

## Submodules

### Current Submodules in Repository

LAT5150DRVMIL contains **2 active submodules:**

#### 1. DIRECTEYE Intelligence Platform
- **Location:** `rag_system/mcp_servers/DIRECTEYE`
- **URL:** `https://github.com/SWORDIntel/DIRECTEYE.git`
- **Status:** ✅ Fully integrated (committed as source files, not git submodule)
- **Size:** 22,403 files
- **Version:** 6.0.0
- **Purpose:** Enterprise OSINT, blockchain, and threat intelligence

**Components:**
```
DIRECTEYE/
├── directeye_main.py          # Unified entry point
├── backend/                    # FastAPI backend (port 8000)
├── core/                       # Core functionality
│   ├── mcp_server/            # MCP server integration
│   └── nl_processing/         # Natural language processing
├── us_lookup/mcp_server/      # OSINT services
│   └── services/              # 40+ intelligence services
├── agents/                     # AI agent systems
├── Addons/binary/             # AVX2/AVX512 native optimizations
├── config/                     # Configuration files
├── data/                       # Data storage
└── cockpit/                    # Dashboard interface
```

#### 2. NCS2 Driver (Neural Compute Stick 2)
- **Location:** `04-hardware/ncs2-driver`
- **URL:** `https://github.com/SWORDIntel/NUC2.1`
- **Status:** ✅ Initialized (active submodule)
- **Commit:** `77eaaeefa519d503302b0939893f90af6f897957`
- **Branch:** `movidius-x-vpu-driver`
- **Purpose:** Intel Movidius VPU driver for hardware acceleration

**Components:**
```
ncs2-driver/
├── movidius_x_vpu.c          # Kernel driver
├── movidius-rs/              # Rust bindings
├── docs/                     # Documentation
├── Dockerfile                # Container support
└── README.md                 # Integration guide
```

### Removed/Cleaned Submodules

The following orphaned submodules were removed during integration:
- `00-documentation/General_Knowledge/Malware_Analysis/awesome-malware-analysis`
- `00-documentation/General_Knowledge/SIGINT_OSINT/awesome-osint`
- `00-documentation/General_Knowledge/Security/Infosec_Reference`
- `00-documentation/General_Knowledge/Security/awesome-cve-poc`
- `00-documentation/General_Knowledge/Security/awesome-hacking`
- `00-documentation/General_Knowledge/Security/awesome-infosec`
- `00-documentation/General_Knowledge/Security/awesome-security`
- `00-documentation/General_Knowledge/Security/h4cker`
- `00-documentation/General_Knowledge/Threat_Intelligence/awesome-threat-intelligence`
- `00-documentation/Security_Feed/VX_Underground/vxug_git_clone`

---

## Integration Points

### 1. AI Engine Integration

**Primary Wrapper:** `ai_engine/directeye_intelligence.py`

This module provides seamless access to DIRECTEYE from the AI engine:

```python
from LAT5150DRVMIL.ai_engine import DirectEyeIntelligence

# Initialize
intel = DirectEyeIntelligence()

# OSINT Query
results = await intel.osint_query("target@example.com")

# Blockchain Analysis
blockchain_info = await intel.blockchain_analyze("0xAddress...", chain="ethereum")

# Threat Intelligence
threat_data = await intel.threat_intelligence("1.1.1.1", indicator_type="ip")

# Get available services
services = intel.get_available_services()

# CPU capabilities (AVX512/AVX2)
caps = intel.cpu_capabilities
```

**Features:**
- Lazy-loading of DIRECTEYE components
- Automatic delegation to full integration (06-intel-systems)
- Graceful fallback to direct access
- CPU capability detection (AVX512/AVX2/SSE)
- Service orchestration (backend API, MCP server)

### 2. Intelligence Systems Integration

**Full Integration:** `06-intel-systems/directeye_intelligence_integration.py`

This is the comprehensive integration with all 35 MCP tools:

```python
from directeye_intelligence_integration import initialize_directeye

# Initialize
intel = initialize_directeye()

# Get capabilities
caps = intel.get_capabilities()

# List all tools
tools = intel.list_tools()  # 35 tools

# OSINT
await intel.osint_query("target")
await intel.breach_data_check("email@example.com")
await intel.corporate_intelligence("Company Name")

# Blockchain
await intel.analyze_blockchain_address("address", "bitcoin")
await intel.check_sanctions("address")
await intel.track_transaction_chain("tx_hash", max_hops=5)

# ML Analytics
await intel.ml_risk_score("address")
await intel.resolve_entity("entity_name")
await intel.predict_risk_trajectory("address", horizon=7)

# Query Chaining
await intel.chain_person_to_crypto("John Doe", location="NYC")
await intel.chain_corporate_investigation("Company")
await intel.chain_email_investigation("email@example.com")
```

### 3. Direct DIRECTEYE Access

**Entry Point:** `rag_system/mcp_servers/DIRECTEYE/directeye_main.py`

Direct access to DIRECTEYE orchestration:

```python
from directeye_main import DirectEyeOrchestrator, CPUDetector

# CPU Detection
detector = CPUDetector()
print(detector.capabilities.cpu_arch)       # AVX512/AVX2/SSE
print(detector.capabilities.avx512_available)
print(detector.capabilities.p_cores)

# Service Orchestration
orchestrator = DirectEyeOrchestrator()
orchestrator.services['backend_api'].auto_start = True
orchestrator.services['mcp_server'].auto_start = True

# Start services
await orchestrator.start_all(enable_simd=True)

# Check status
status = await orchestrator.check_service_health('backend_api')
await orchestrator.print_status()
```

**CLI Usage:**
```bash
# Start all services
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py start --all

# Check capabilities
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py capabilities

# Check status
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py status

# Show platform info
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py info
```

---

## DIRECTEYE Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LAT5150DRVMIL System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────┐         ┌──────────────────────┐   │
│  │   AI Engine       │────────▶│  DIRECTEYE           │   │
│  │   (02-ai-engine)  │         │  Intelligence        │   │
│  └───────────────────┘         └──────────────────────┘   │
│           │                              │                 │
│           │                              ▼                 │
│           │                    ┌──────────────────────┐   │
│           │                    │  Intelligence Systems│   │
│           │                    │  (06-intel-systems)  │   │
│           │                    └──────────────────────┘   │
│           │                              │                 │
│           ▼                              ▼                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │         DIRECTEYE Platform                          │  │
│  │   (rag_system/mcp_servers/DIRECTEYE)                │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │ Backend API  │  │  MCP Server  │  │ OSINT    │ │  │
│  │  │  (Port 8000) │  │  (Port 8001) │  │ Services │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │  │
│  │                                                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │ Blockchain   │  │ ML Analytics │  │ Native   │ │  │
│  │  │ Intel (12+)  │  │ (5 engines)  │  │ SIMD     │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │  │
│  │                                                     │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │    35+ MCP Tools (AI-Powered)               │  │  │
│  │  │  • Entity Resolution  • Risk Scoring        │  │  │
│  │  │  • Query Chaining     • Threat Intel        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                               │
│                           ▼                               │
│              ┌─────────────────────────┐                 │
│              │  PostgreSQL / Redis     │                 │
│              │  Neo4j (optional)       │                 │
│              └─────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User/AI Request
    ↓
AI Engine (DSMILAIEngine)
    ↓
DirectEyeIntelligence Wrapper
    ↓
06-intel-systems Integration (if available)
    ↓
DIRECTEYE Platform
    ↓
┌──────────┬─────────────┬──────────────┬─────────────┐
│ OSINT    │ Blockchain  │ ML Analytics │ MCP Server  │
│ Services │ Intelligence│              │             │
└────┬─────┴──────┬──────┴──────┬───────┴──────┬──────┘
     │            │             │              │
     ▼            ▼             ▼              ▼
 40+ Data     12+ Chains    5 Engines    35+ Tools
 Sources
     │            │             │              │
     └────────────┴─────────────┴──────────────┘
                      │
                      ▼
           PostgreSQL / Redis / Neo4j
                      │
                      ▼
              Results returned to AI
```

---

## Usage Examples

### Example 1: OSINT Investigation

```python
from LAT5150DRVMIL.ai_engine import DirectEyeIntelligence

async def investigate_person():
    intel = DirectEyeIntelligence()

    # Search across multiple sources
    results = await intel.osint_query("John Doe NYC")

    # Check for breaches
    breach_info = await intel.osint_query(
        "check john.doe@example.com for breaches"
    )

    # Corporate intelligence
    company_data = await intel.osint_query(
        "search SEC filings for Acme Corp"
    )

    return {
        "person": results,
        "breaches": breach_info,
        "corporate": company_data
    }
```

### Example 2: Blockchain Analysis

```python
async def analyze_crypto_wallet():
    intel = DirectEyeIntelligence()

    address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"

    # Analyze address
    analysis = await intel.blockchain_analyze(address, chain="bitcoin")

    # Check sanctions
    sanctions = await intel.blockchain_analyze(
        address,
        chain="bitcoin"
    )

    return {
        "analysis": analysis,
        "sanctions": sanctions
    }
```

### Example 3: AI Engine with Intelligence

```python
from LAT5150DRVMIL.ai_engine import DSMILAIEngine, DirectEyeIntelligence

async def ai_with_intelligence():
    # Initialize both engines
    ai = DSMILAIEngine()
    intel = DirectEyeIntelligence()

    # User query
    user_query = "Investigate email john@example.com"

    # Get intelligence
    osint_results = await intel.osint_query(user_query)

    # Generate AI response with context
    ai_response = ai.generate_with_rag(
        prompt=user_query,
        context=osint_results
    )

    return {
        "intelligence": osint_results,
        "ai_response": ai_response
    }
```

### Example 4: Start DIRECTEYE Services

```python
async def start_intelligence_platform():
    intel = DirectEyeIntelligence()

    # Start backend API and MCP server with SIMD optimization
    await intel.start_services(
        backend=True,
        mcp_server=True,
        enable_simd=True
    )

    # Check status
    status = intel.get_service_status()
    print(f"Backend API: {status['backend_api']['status']}")
    print(f"MCP Server: {status['mcp_server']['status']}")
    print(f"CPU: {status['cpu_capabilities']['arch']}")

    # Get available services
    services = intel.get_available_services()
    print(f"OSINT Services: {len(services['osint'])}")
    print(f"Blockchain Chains: {len(services['blockchain'])}")
```

---

## Service Inventory

### OSINT Services (40+)

#### People Search (3)
- TruePeopleSearch
- Hunter.io (email finding/verification)
- EmailRep (reputation scoring)

#### Breach Data (5)
- Have I Been Pwned (HIBP)
- SpyCloud (50M+ records local DB)
- Snusbase (API active)
- LeakOSINT (API active)
- HackCheck (email breach verification)

#### Corporate Intelligence (6)
- SEC EDGAR (US company filings)
- Companies House UK
- ICIJ Offshore Leaks (with Neo4j option)
- Corporate registries
- Officer searches
- Financial filings

#### Government Data (4 platforms, 398+ endpoints)
- Data.gov (US federal datasets)
- SOCRATA (26+ state portals, Florida keys)
- CKAN (worldwide government portals)
- DKAN (healthcare/CMS data)

#### Threat Intelligence (4)
- AlienVault OTX (needs key)
- Censys (network intelligence)
- FOFA (cyberspace search)
- IPGeolocation

#### Additional Services (3+)
- Doxbin Archive
- Hugging Face (AI/ML models)
- TRM Labs (blockchain intelligence)

### Blockchain Intelligence (12+ Chains)

- Bitcoin
- Ethereum
- Polygon
- Avalanche
- Arbitrum
- Optimism
- Base
- Binance Smart Chain
- Fantom
- Cronos
- Gnosis
- Moonbeam

**Features:**
- 100K+ labeled addresses
- Entity attribution
- Sanctions screening (OFAC/UN/EU)
- Transaction tracking
- Risk assessment
- Multi-hop analysis

### MCP Tools (35+)

#### Core Government Data (8)
- search_entities
- get_entity_details
- get_entity_relationships
- get_entities_by_type
- get_system_statistics
- get_datasets
- export_entities
- health_check

#### OSINT (3)
- osint_query
- osint_help
- osint_status

#### Blockchain Analysis (7)
- blockchain_analyze_address
- blockchain_get_transaction
- crypto_market_data
- blockchain_risk_assessment
- comprehensive_blockchain_analysis
- multi_chain_search
- blockchain_services_health

#### Entity Attribution & Chain Tracking (6)
- identify_entity
- check_sanctions
- track_transaction_chain
- search_entities_by_type
- get_entity_risk_profile
- batch_identify_addresses

#### ML Analytics (6)
- ml_score_transaction_risk
- resolve_entity
- predict_risk_trajectory
- analyze_cross_chain_entity
- analyze_transaction_network
- get_early_warnings

#### Query Chaining (5)
- execute_query_chain
- chain_person_to_crypto
- chain_crypto_tracking
- chain_corporate_investigation
- chain_email_investigation

### ML Analytics Engines (5)

1. **Risk Scoring Engine**
   - Random Forest
   - XGBoost
   - Isolation Forest
   - 28-feature engineering

2. **Entity Resolution Engine**
   - Levenshtein distance
   - Jaro-Winkler similarity
   - Cosine similarity
   - Phonetic matching

3. **Predictive Analytics Engine**
   - 1/7/30 day horizons
   - Early warning system
   - Trend analysis

4. **Cross-Chain Analysis Engine**
   - 12+ blockchain support
   - Entity attribution
   - Network analysis

5. **Graph Analytics Engine**
   - Centrality measures
   - Community detection
   - Key player identification

---

## File Locations

### Core Integration Files

```
LAT5150DRVMIL/
│
├── ai_engine/
│   ├── __init__.py                              # Updated with DIRECTEYE
│   └── directeye_intelligence.py                # AI Engine wrapper [NEW]
│
├── 06-intel-systems/
│   └── directeye_intelligence_integration.py    # Full integration [EXISTS]
│
├── rag_system/mcp_servers/DIRECTEYE/            # DIRECTEYE source [22,403 files]
│   ├── directeye_main.py                        # Entry point
│   ├── backend/                                 # FastAPI backend
│   ├── core/mcp_server/                         # MCP server
│   ├── us_lookup/mcp_server/services/          # OSINT services
│   ├── agents/                                  # AI agents
│   ├── Addons/binary/                           # Native SIMD
│   └── config/                                  # Configuration
│
├── 04-hardware/ncs2-driver/                     # NCS2 submodule [ACTIVE]
│
├── .gitmodules                                  # Submodule configuration
│
└── DIRECTEYE_FULL_INTEGRATION.md               # This file [NEW]
```

### Documentation Files

```
rag_system/mcp_servers/DIRECTEYE/
├── DIRECTEYE_ENTRY_POINT.md              # Entry point guide (8,000+ words)
├── MODULE_INTEGRATION_SUMMARY.md          # Integration summary (6,000+ words)
├── SUBMODULE_INTEGRATION.md               # Submodule guide
├── API_DOCUMENTATION.md                   # API reference
├── CLAUDE.md                              # Claude AI instructions
└── PRODUCTION_DEPLOYMENT_GUIDE.md         # Deployment guide
```

---

## Quick Start

### 1. From AI Engine

```python
from LAT5150DRVMIL.ai_engine import DirectEyeIntelligence

intel = DirectEyeIntelligence()

# Check capabilities
print(intel.cpu_capabilities)

# OSINT query
results = await intel.osint_query("target@example.com")
```

### 2. From Intelligence Systems

```python
from directeye_intelligence_integration import initialize_directeye

intel = initialize_directeye()

# List all 35 tools
tools = intel.list_tools()

# Use tools
await intel.breach_data_check("email@example.com")
await intel.analyze_blockchain_address("address")
```

### 3. Direct DIRECTEYE

```bash
# CLI
cd rag_system/mcp_servers/DIRECTEYE
python directeye_main.py start --all

# Python
from directeye_main import DirectEyeOrchestrator
orchestrator = DirectEyeOrchestrator()
await orchestrator.start_all()
```

### 4. Check System Status

```python
intel = DirectEyeIntelligence()

# CPU capabilities
caps = intel.cpu_capabilities
print(f"Architecture: {caps['arch']}")
print(f"AVX-512: {caps['avx512']}")
print(f"Cores: {caps['cores']}")

# Service status
status = intel.get_service_status()

# Available services
services = intel.get_available_services()
```

---

## Performance

### SIMD Optimization

| Operation | Software | AVX2 | AVX-512 |
|-----------|----------|------|---------|
| Packet Encode | 350 ns | 71 ns | 42 ns |
| Packet Decode | 320 ns | 66 ns | 39 ns |
| Batch Process | 280 ns | 52 ns | 28 ns |
| **Speedup** | 1x | **5x** | **8-12x** |

### Service Response Times

| Endpoint | Response Time | Throughput |
|----------|---------------|------------|
| `/health` | <15ms | 10,000 req/s |
| OSINT Query | <200ms | 500 req/s |
| Blockchain Analysis | <300ms | 300 req/s |
| ML Risk Scoring | <100ms | 1,000 req/s |

---

## Configuration

### Environment Variables

```bash
# DIRECTEYE Configuration
export DIRECTEYE_ENV=production
export DIRECTEYE_LOG_LEVEL=INFO

# Database
export DATABASE_URL=postgresql://directeye_user:directeye2025@localhost:5432/directeye
export REDIS_URL=redis://localhost:6379

# Hardware Optimization
export PHOENIX_FORCE_AVX512=1          # Force AVX-512
export PHOENIX_P_CORES=0,1,2,3         # P-core IDs
export PHOENIX_DEBUG=1                 # Debug mode

# Security
export JWT_SECRET_KEY=your-secret-key
export VAULT_PASSWORD=directeye2025
```

### Configuration Files

**Main Config:** `rag_system/mcp_servers/DIRECTEYE/config/directeye_config.yaml`

```yaml
hardware:
  cpu:
    enable_simd: true
    force_avx512: false
    p_cores: []  # Auto-detect

services:
  backend_api:
    enabled: true
    port: 8000
    workers: 4

  mcp_server:
    enabled: true
    port: 8001
    workers: 2

database:
  postgresql:
    host: localhost
    port: 5432
    database: directeye
```

---

## Support & Resources

### Documentation
- **Entry Point Guide:** `rag_system/mcp_servers/DIRECTEYE/DIRECTEYE_ENTRY_POINT.md`
- **Integration Summary:** `rag_system/mcp_servers/DIRECTEYE/MODULE_INTEGRATION_SUMMARY.md`
- **API Docs:** `rag_system/mcp_servers/DIRECTEYE/API_DOCUMENTATION.md`

### GitHub
- **DIRECTEYE:** https://github.com/SWORDIntel/DIRECTEYE
- **NCS2 Driver:** https://github.com/SWORDIntel/NUC2.1

### Contact
- **Issues:** https://github.com/SWORDIntel/DIRECTEYE/issues
- **Email:** support@swordint.com

---

## Version Information

- **DIRECTEYE:** 6.0.0 (Production Ready)
- **LAT5150DRVMIL:** Current
- **Integration Status:** ✅ Complete
- **Last Updated:** November 12, 2025

---

**© 2025 SWORD Intelligence**
