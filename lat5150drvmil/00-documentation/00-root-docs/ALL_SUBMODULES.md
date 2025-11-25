# LAT5150DRVMIL - Complete Submodules List

**Last Updated:** November 12, 2025
**Branch:** `claude/integrate-directeye-submodules-011CV3QzaFSjUemsNbrrY3XF`

---

## Active Submodules (2)

### 1. DIRECTEYE - Enterprise Intelligence Platform ⭐

**Location:** `rag_system/mcp_servers/DIRECTEYE`
**Repository:** https://github.com/SWORDIntel/DIRECTEYE.git
**Version:** 6.0.0
**Status:** ✅ Fully Integrated (committed as source files)
**Size:** 22,403 files

#### Purpose
Comprehensive OSINT, blockchain, and threat intelligence platform with:
- 40+ OSINT services (people search, breach data, corporate intel, threat feeds, government data)
- 12+ blockchain networks (Bitcoin, Ethereum, Polygon, etc.)
- 35+ AI-powered MCP tools
- 5 ML analytics engines
- Native AVX2/AVX512 optimization

#### Integration Points
1. **AI Engine:** `ai_engine/directeye_intelligence.py`
2. **Intelligence Systems:** `06-intel-systems/directeye_intelligence_integration.py`
3. **Direct Access:** `rag_system/mcp_servers/DIRECTEYE/directeye_main.py`

#### Key Features
- Backend FastAPI server (port 8000)
- MCP server with 35+ tools (port 8001)
- Post-quantum cryptography (ML-KEM-1024)
- Hardware acceleration (AVX512 → AVX2 → SSE4.2)
- P-core/E-core detection and affinity
- Real-time intelligence correlation
- Multi-step query chaining

#### Services Breakdown
```
OSINT Services (40+):
  ├── People Search (3): TruePeopleSearch, Hunter.io, EmailRep
  ├── Breach Data (5): HIBP, SpyCloud, Snusbase, LeakOSINT, HackCheck
  ├── Corporate (6): SEC EDGAR, Companies House, ICIJ, etc.
  ├── Government (4 platforms, 398+ endpoints): Data.gov, SOCRATA, CKAN, DKAN
  ├── Threat Intel (4): AlienVault OTX, Censys, FOFA, IPGeolocation
  └── Additional (3+): Doxbin, Hugging Face, TRM Labs

Blockchain Intelligence (12+ chains):
  ├── Bitcoin, Ethereum, Polygon, Avalanche
  ├── Arbitrum, Optimism, Base, BSC
  ├── Fantom, Cronos, Gnosis, Moonbeam
  └── Features:
      ├── 100K+ labeled addresses
      ├── Entity attribution
      ├── Sanctions screening (OFAC/UN/EU)
      ├── Transaction tracking
      └── Risk assessment

MCP Tools (35+):
  ├── Core Government Data (8 tools)
  ├── OSINT (3 tools)
  ├── Blockchain Analysis (7 tools)
  ├── Entity Attribution (6 tools)
  ├── ML Analytics (6 tools)
  └── Query Chaining (5 tools)

ML Analytics (5 engines):
  ├── Risk Scoring (Random Forest, XGBoost, Isolation Forest)
  ├── Entity Resolution (Levenshtein, Jaro-Winkler, Cosine, Phonetic)
  ├── Predictive Analytics (1/7/30 day horizons)
  ├── Cross-Chain Analysis (12+ blockchains)
  └── Graph Analytics (Centrality, Communities, Key Players)
```

#### Usage Examples
```python
# From AI Engine
from LAT5150DRVMIL.ai_engine import DirectEyeIntelligence

intel = DirectEyeIntelligence()
results = await intel.osint_query("target@example.com")
blockchain = await intel.blockchain_analyze("0xAddress...")

# CLI
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py start --all
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py capabilities
```

---

### 2. NCS2 Driver - Neural Compute Stick 2

**Location:** `04-hardware/ncs2-driver`
**Repository:** https://github.com/SWORDIntel/NUC2.1
**Commit:** `77eaaeefa519d503302b0939893f90af6f897957`
**Branch:** `movidius-x-vpu-driver`
**Status:** ✅ Active Submodule (initialized)

#### Purpose
Intel Movidius Myriad X VPU driver for hardware-accelerated AI inference on Neural Compute Stick 2 (NCS2).

#### Features
- Kernel-level VPU driver (`movidius_x_vpu.c`)
- Rust bindings (`movidius-rs/`)
- Docker containerization support
- NCAPI v2 integration
- Performance optimization for WhiteRabbitNeo

#### Integration
- **Kernel Module:** Loaded at boot via DKMS
- **AI Engine:** Used by `02-ai-engine/whiterabbitneo_inference_engine.py`
- **Documentation:** `04-hardware/NCS2_INTEGRATION.md`

#### Hardware Requirements
- Intel NCS2 device
- USB 3.0 port
- Intel Movidius Myriad X VPU

---

## Submodule Summary

| #   | Name | Location | Status | Purpose |
|-----|------|----------|--------|---------|
| 1   | **DIRECTEYE** | `rag_system/mcp_servers/DIRECTEYE` | ✅ Integrated | OSINT/Blockchain/Threat Intel (40+ services, 35+ tools) |
| 2   | **NCS2 Driver** | `04-hardware/ncs2-driver` | ✅ Active | Intel Movidius VPU driver for hardware acceleration |

**Total Active Submodules:** 2

---

## Removed/Cleaned Submodules

The following submodules were removed during repository cleanup (orphaned, not in .gitmodules):

| # | Path | Reason |
|---|------|--------|
| 1 | `00-documentation/General_Knowledge/Malware_Analysis/awesome-malware-analysis` | Orphaned (not in .gitmodules) |
| 2 | `00-documentation/General_Knowledge/SIGINT_OSINT/awesome-osint` | Orphaned (not in .gitmodules) |
| 3 | `00-documentation/General_Knowledge/Security/Infosec_Reference` | Orphaned (not in .gitmodules) |
| 4 | `00-documentation/General_Knowledge/Security/awesome-cve-poc` | Orphaned (not in .gitmodules) |
| 5 | `00-documentation/General_Knowledge/Security/awesome-hacking` | Orphaned (not in .gitmodules) |
| 6 | `00-documentation/General_Knowledge/Security/awesome-infosec` | Orphaned (not in .gitmodules) |
| 7 | `00-documentation/General_Knowledge/Security/awesome-security` | Orphaned (not in .gitmodules) |
| 8 | `00-documentation/General_Knowledge/Security/h4cker` | Orphaned (not in .gitmodules) |
| 9 | `00-documentation/General_Knowledge/Threat_Intelligence/awesome-threat-intelligence` | Orphaned (not in .gitmodules) |
| 10 | `00-documentation/Security_Feed/VX_Underground/vxug_git_clone` | Orphaned (not in .gitmodules) |

**Total Removed:** 10 orphaned submodules

---

## Integration Architecture

```
LAT5150DRVMIL/
│
├── ai_engine/                                   # AI Engine Package
│   ├── __init__.py                              # Exports DirectEyeIntelligence
│   └── directeye_intelligence.py                # DIRECTEYE wrapper [NEW]
│
├── 02-ai-engine/                                # Main AI Engine
│   ├── dsmil_ai_engine.py                       # 5 models via Ollama
│   ├── whiterabbitneo_inference_engine.py       # Uses NCS2
│   └── ...
│
├── 06-intel-systems/                            # Intelligence Integration
│   └── directeye_intelligence_integration.py    # Full DIRECTEYE integration
│
├── rag_system/mcp_servers/DIRECTEYE/           # DIRECTEYE Platform ⭐
│   ├── directeye_main.py                        # Entry point (800+ lines)
│   ├── backend/                                 # FastAPI (port 8000)
│   │   ├── api/                                 # REST API endpoints
│   │   ├── auth/                                # JWT authentication
│   │   ├── security/                            # Security middleware
│   │   └── analytics/                           # ML analytics
│   ├── core/                                    # Core functionality
│   │   ├── mcp_server/                          # MCP integration
│   │   ├── nl_processing/                       # NLP engine
│   │   └── state_handlers/                      # State management
│   ├── us_lookup/mcp_server/                    # OSINT services
│   │   └── services/                            # 40+ intelligence services
│   ├── agents/                                  # AI agent systems
│   │   ├── enumeration/                         # Discovery agents
│   │   ├── orchestration/                       # Coordination
│   │   └── verification/                        # Truth validation
│   ├── Addons/binary/                           # Native optimizations
│   │   └── native/                              # AVX2/AVX512 SIMD
│   ├── config/                                  # Configuration
│   │   ├── json/                                # JSON configs
│   │   ├── mcp/                                 # MCP configs
│   │   └── directeye_config.yaml               # Main config
│   ├── data/                                    # Data storage
│   ├── cockpit/                                 # Dashboard
│   └── docs/                                    # Documentation
│
├── 04-hardware/ncs2-driver/                    # NCS2 Submodule ⭐
│   ├── movidius_x_vpu.c                         # Kernel driver
│   ├── movidius-rs/                             # Rust bindings
│   └── docs/                                    # Documentation
│
├── .gitmodules                                  # Submodule configuration
│
└── Documentation/                               # Integration docs [NEW]
    ├── DIRECTEYE_FULL_INTEGRATION.md            # Complete integration guide
    └── ALL_SUBMODULES.md                        # This file
```

---

## Quick Commands

### Check Submodules
```bash
# List all submodules
git submodule status

# Show submodule configuration
cat .gitmodules

# Update submodules
git submodule update --init --recursive
```

### DIRECTEYE Operations
```bash
# Start DIRECTEYE services
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py start --all

# Check capabilities
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py capabilities

# Service status
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py status

# Platform info
python rag_system/mcp_servers/DIRECTEYE/directeye_main.py info
```

### Python Integration
```python
# AI Engine with DIRECTEYE
from LAT5150DRVMIL.ai_engine import DSMILAIEngine, DirectEyeIntelligence

ai = DSMILAIEngine()
intel = DirectEyeIntelligence()

# Intelligence operations
await intel.osint_query("target")
await intel.blockchain_analyze("address")
await intel.threat_intelligence("1.1.1.1")
```

---

## Performance Metrics

### DIRECTEYE Performance

**SIMD Optimization:**
- AVX-512: 8-16x speedup vs software
- AVX2: 4-8x speedup vs software
- Packet encoding: 42ns (AVX-512), 71ns (AVX2), 350ns (software)

**Service Response Times:**
- Health check: <15ms
- OSINT query: <200ms
- Blockchain analysis: <300ms
- ML risk scoring: <100ms

**Throughput:**
- Health endpoint: 10,000 req/s
- OSINT queries: 500 req/s
- Blockchain analysis: 300 req/s
- ML operations: 1,000 req/s

---

## Configuration

### Environment Variables
```bash
# DIRECTEYE
export DIRECTEYE_ENV=production
export DIRECTEYE_LOG_LEVEL=INFO
export DATABASE_URL=postgresql://directeye_user:directeye2025@localhost:5432/directeye
export REDIS_URL=redis://localhost:6379

# Hardware
export PHOENIX_FORCE_AVX512=1
export PHOENIX_P_CORES=0,1,2,3
export PHOENIX_DEBUG=1

# Security
export JWT_SECRET_KEY=your-secret-key
export VAULT_PASSWORD=directeye2025
```

### Git Configuration
```bash
# .gitmodules content
[submodule "rag_system/mcp_servers/DIRECTEYE"]
    path = rag_system/mcp_servers/DIRECTEYE
    url = https://github.com/SWORDIntel/DIRECTEYE.git

[submodule "04-hardware/ncs2-driver"]
    path = 04-hardware/ncs2-driver
    url = https://github.com/SWORDIntel/NUC2.1
```

---

## Support & Resources

### Documentation
- **Integration Guide:** `DIRECTEYE_FULL_INTEGRATION.md`
- **DIRECTEYE Entry Point:** `rag_system/mcp_servers/DIRECTEYE/DIRECTEYE_ENTRY_POINT.md`
- **Module Summary:** `rag_system/mcp_servers/DIRECTEYE/MODULE_INTEGRATION_SUMMARY.md`
- **NCS2 Guide:** `04-hardware/NCS2_INTEGRATION.md`

### GitHub Repositories
- **LAT5150DRVMIL:** https://github.com/SWORDIntel/LAT5150DRVMIL
- **DIRECTEYE:** https://github.com/SWORDIntel/DIRECTEYE
- **NCS2 Driver:** https://github.com/SWORDIntel/NUC2.1

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-12 | 1.0.0 | Initial integration - DIRECTEYE fully integrated into AI engine |
| 2025-11-12 | 1.0.0 | Cleaned 10 orphaned submodules |
| 2025-11-12 | 1.0.0 | Initialized NCS2 driver submodule |
| 2025-11-12 | 1.0.0 | Created comprehensive documentation |

---

**Status:** ✅ **Complete - All Submodules Integrated and Documented**
**Branch:** `claude/integrate-directeye-submodules-011CV3QzaFSjUemsNbrrY3XF`
**© 2025 SWORD Intelligence**
