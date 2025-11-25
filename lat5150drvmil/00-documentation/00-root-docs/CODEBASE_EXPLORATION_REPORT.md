# LAT5150DRVMIL - Comprehensive Codebase Exploration Report

**Date:** November 12, 2025  
**Project:** Dell Latitude 5450 Covert AI Platform (LAT5150DRVMIL)  
**Current Branch:** claude/integrate-directeye-mcp-011CV3PR4xnXPejgvh3pgNEb  
**Status:** Production Ready (Version 8.3.2)

---

## Executive Summary

LAT5150DRVMIL is a **complete LOCAL-FIRST AI platform** built on Dell MIL-SPEC hardware with:
- **84 DSMIL hardware devices** (656 operations, 79 usable, 5 quarantined)
- **Multi-model AI engine** (5 models with smart routing)
- **Post-Quantum Cryptography** (CSNA 2.0 compliant)
- **12 MCP servers** (including DIRECTEYE blockchain intelligence integration)
- **Screenshot Intelligence System** with Vector RAG (Qdrant)
- **SHRINK submodule** for storage optimization and compression
- **Unified Web Dashboard** (localhost:5050)

---

## 1. DIRECTEYE Integration Status

### Current State

**DIRECTEYE is configured as a Git submodule but not yet initialized:**

```
Submodule Path: rag_system/mcp_servers/DIRECTEYE
Repository URL: https://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@github.com/SWORDIntel/DIRECTEYE.git
Status: Defined in .gitmodules but directory is empty
```

### Integration Points

DIRECTEYE is intended to provide:
- **40+ blockchain intelligence services**
- **35 MCP tools** for blockchain analysis and intelligence
- Location in: `rag_system/mcp_servers/DIRECTEYE/`

### Documentation References
- File: `/home/user/LAT5150DRVMIL/00-documentation/historical-docs/old-status/README_OSINT_INTEGRATION_COMPLETE.md`
- References DIRECTEYE as part of the comprehensive OSINT integration (167+ total OSINT sources)

### Next Steps for DIRECTEYE Integration
```bash
# Initialize the DIRECTEYE submodule
git submodule update --init --recursive rag_system/mcp_servers/DIRECTEYE

# After initialization, add to MCP servers config
# Location: /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json
```

---

## 2. MCP (Model Context Protocol) Server Setup

### Current MCP Configuration

**Location:** `/home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json`

**12 MCP Servers Currently Configured:**

#### Core Python Servers (Ready, No External Setup)
1. **dsmil-ai** - DSMIL AI Engine with RAG, 5 models, PQC status
2. **sequential-thinking** - Multi-step structured reasoning
3. **filesystem** - Sandboxed file operations (read/write/list/search)
4. **memory** - Persistent knowledge graph
5. **fetch** - Web content fetching with SSRF protection
6. **git** - Git operations with command injection protection
7. **screenshot-intelligence** - Screenshot analysis, OCR, timeline analysis

#### External MCP Servers (Require Installation)
8. **search-tools-mcp** - Advanced code search with CodeRank algorithm
9. **docs-mcp-server** - Documentation indexing and vector embeddings
10. **metasploit** - Security testing framework integration
11. **maigret** - Username OSINT across social networks
12. **security-tools** - 23 security tools (Nmap, Nuclei, SQLmap, etc.)

### Security Features

All MCP servers implement:
- **Token-based authentication** (SHA-256 hashing)
- **Rate limiting** (60 req/min per client)
- **Input validation & sanitization**
- **Audit logging** (~/.dsmil/mcp_audit.log)
- **Path traversal prevention**
- **Sandboxing for file operations**

**Security Config File:** `/home/user/LAT5150DRVMIL/02-ai-engine/mcp_security.py`

### Setup Commands

```bash
# Install all MCP servers
cd /home/user/LAT5150DRVMIL
./setup-mcp-servers.sh  # Takes 5-10 minutes

# Verify configuration
cat /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json
```

---

## 3. Database Configuration Files

### PostgreSQL Schema

**Location:** `/home/user/LAT5150DRVMIL/02-ai-engine/database_schema.sql`

**Tables & Features:**

#### Core Tables
- **users** - User accounts and profiles (UUID-based)
- **user_preferences** - User-specific settings (JSONB)
- **conversations** - Conversation sessions (archived support)
- **messages** - Individual messages with model/token/latency tracking

#### RAG System Tables
- **rag_documents** - Document collection with hash deduplication
- **rag_chunks** - Document chunks for granular retrieval
- **rag_chunk_embeddings** - Vector embeddings (384-dimensional, sentence-transformers)
- **rag_retrievals** - Tracking of retrieved chunks per query

#### Knowledge Graph
- **kg_entities** - Named entities with types
- **kg_observations** - Entity observations with confidence scores
- **kg_relations** - Entity relationships with metadata

#### Analytics & Cache
- **query_analytics** - Query performance metrics
- **response_cache** - Persistent cache (with TTL support)
- **model_metrics** - Model performance tracking

#### Memory System (Hierarchical)
- **memory_blocks** - Tiered memory (working/short-term/long-term)
- Support for block prioritization and conversation context

### Advanced Features
- **Vector similarity search** for embeddings
- **Message embeddings** for semantic search across conversations
- **Cache management** with automatic expiration
- **Comprehensive indexing** for performance (34+ indexes)
- **Views** for common queries (recent_conversations, model_performance)
- **Triggers** for automatic timestamp updates
- **Functions** for cleanup and archival

### Archive Database

**Legacy SQLite:** `/home/user/LAT5150DRVMIL/99-archive/database/data/dsmil_tokens.db`

---

## 4. Docker/Container Deployment Files

### Current State

**No Docker/Dockerfile found in main directories.**

However, optional Docker components exist:
- **MCP Docker Setup:** `/home/user/LAT5150DRVMIL/03-mcp-servers/setup_mcp_docker.sh`
  - Provides Docker deployment option for MCP servers
  - Includes Docker Compose configuration options

### Deployment Environment

**Environment Config:** `/home/user/LAT5150DRVMIL/05-deployment/npu-covert-edition.env`

Contains:
- System configuration settings
- NPU (Neural Processing Unit) settings for Intel Core Ultra 7
- Deployment parameters for covert edition

### Deployment Structure

```
05-deployment/
├── npu-covert-edition.env              # Environment variables
├── verify_system.sh                    # System verification
├── systemd/                            # systemd service files
└── zfs/                                # ZFS storage integration
```

---

## 5. Service Architecture & Microservices Setup

### Multi-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│    Unified Dashboard (localhost:5050)            │
├─────────────────────────────────────────────────┤
│                                                 │
│  Web Interface (03-web-interface/)              │
│  ├── clean_ui_v3.html (Modern 3-panel UI)      │
│  └── dsmil_unified_server.py (Backend)          │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  AI Engine Layer (02-ai-engine/)                │
│  ├── DSMILAIEngine (5 models)                   │
│  ├── UnifiedAIOrchestrator (multi-backend)      │
│  ├── AgentOrchestrator (25+ agents)             │
│  ├── ACE Framework (Advanced Compute Engine)    │
│  └── Quantum Crypto Layer (CSNA 2.0)            │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Integration Layer (04-integrations/)           │
│  ├── RAG Manager (200+ document KB)             │
│  ├── Screenshot Intelligence System             │
│  ├── Vector RAG (Qdrant + BAAI 88%+ accuracy)   │
│  ├── Web Scraper (crawl4ai integration)         │
│  └── OSINT Collectors (167+ sources)            │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Hardware Layer (02-ai-engine/)                 │
│  ├── DSMIL Subsystem (84 devices, 656 ops)      │
│  ├── TPM 2.0 Integration (88 algorithms)        │
│  ├── NPU Control (Intel AI Boost 3720)          │
│  └── HardwareMCU Interface                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Services

1. **AI Engine Service** - Main inference engine
   - File: `/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_ai_engine.py`
   - Supports: fast, code, quality, uncensored, large models

2. **Unified Orchestrator** - Multi-backend routing
   - File: `/home/user/LAT5150DRVMIL/02-ai-engine/unified_orchestrator.py`
   - Routes to: Local (Ollama), Gemini, OpenAI

3. **Screenshot Intelligence** - Image analysis + vector RAG
   - File: `/home/user/LAT5150DRVMIL/screenshot_intel/`
   - Features: OCR (PaddleOCR/Tesseract), timeline analysis, anomaly detection

4. **Agent Orchestrator** - Multi-agent coordination
   - File: `/home/user/LAT5150DRVMIL/02-ai-engine/agent_orchestrator.py`
   - Manages 25+ specialized agents

5. **DSMIL Subsystem Controller** - Hardware control
   - File: `/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_subsystem_controller.py`
   - Controls: 84 devices across 7 groups

### Web Interface

**Main Dashboard:**
- URL: `http://localhost:5050`
- Files: `03-web-interface/clean_ui_v3.html` + `dsmil_unified_server.py`
- Features: 3-panel layout, real-time status, command execution

---

## 6. Submodule Configurations

### Configured Submodules

#### 1. DIRECTEYE (Blockchain Intelligence)
```
Path: rag_system/mcp_servers/DIRECTEYE
URL: https://github.com/SWORDIntel/DIRECTEYE
Status: Defined, not initialized
Purpose: 40+ blockchain intelligence services
```

#### 2. NCS2 Driver (Intel Neural Compute Stick)
```
Path: 04-hardware/ncs2-driver
URL: https://github.com/SWORDIntel/NUC2.1
Status: Defined, not initialized
Purpose: Intel NCS2 accelerator support
```

### SHRINK Submodule (Storage Optimization)

**Not a git submodule but deeply integrated as Python package:**

```
Features:
- Intelligent compression (zstd, lz4, brotli)
- Auto-algorithm selection
- Resource optimization (memory, disk, network)
- Data deduplication (content-addressable storage)
- 60-80% space savings on screenshots

Integration Manager: /home/user/LAT5150DRVMIL/shrink_integration_manager.py
Configuration Guide: /home/user/LAT5150DRVMIL/SHRINK_INTEGRATION.md
Health Monitor: submodule_health_monitor.py
```

### Submodule Initialization

```bash
# Initialize all submodules
git submodule update --init --recursive

# Initialize specific submodule
git submodule update --init --recursive rag_system/mcp_servers/DIRECTEYE

# Update to latest versions
git submodule update --remote
```

---

## 7. Complete Directory Structure

### Top-Level Organization

```
LAT5150DRVMIL/
├── 00-documentation/          # 80+ docs (30+ directories)
│   ├── 00-indexes/            # Navigation guides
│   ├── 00-root-docs/          # DSMIL, SWORD, core refs
│   ├── 01-planning/           # 18 implementation plans
│   ├── 02-analysis/           # Hardware/security analysis
│   ├── 03-ai-framework/       # AI orchestration docs
│   ├── 04-progress/           # Session summaries
│   └── 05-reference/          # Original requirements
│
├── 01-source/                 # DSMIL kernel & framework
│   ├── kernel/                # dsmil-72dev.c (84 devices)
│   │   ├── build-and-install.sh
│   │   └── rust/              # Safety layer (10,280 lines)
│   └── scripts/               # Hardware utilities
│
├── 02-ai-engine/              # Core AI platform (250+ Python files)
│   ├── dsmil_ai_engine.py     # 5 models (fast/code/quality/uncensored/large)
│   ├── unified_orchestrator.py # Multi-backend routing
│   ├── agent_orchestrator.py  # 25+ agent coordination
│   ├── dsmil_subsystem_controller.py # 84 devices
│   ├── quantum_crypto_layer.py # CSNA 2.0 PQC
│   ├── tpm_crypto_integration.py # TPM 2.0 (88 algos)
│   ├── ai_gui_dashboard.py    # Main GUI entry point
│   ├── mcp_servers_config.json # 12 MCP server configs
│   ├── mcp_security.py        # Auth/rate-limiting/audit
│   ├── database_schema.sql    # PostgreSQL schema
│   ├── ai_benchmarking.py     # 22 comprehensive tests
│   ├── dsmil_guided_activation.py # Device activation TUI
│   ├── dsmil_operation_monitor.py # 656 operations browser
│   └── ace_*.py               # ACE Framework (advanced compute)
│
├── 02-tools/                  # Specialized tools
│
├── 03-mcp-servers/            # 12 MCP server integrations
│   ├── setup_mcp_servers.sh   # Auto-installer
│   ├── setup_mcp_docker.sh    # Docker deployment
│   ├── gemini/                # Google Gemini MCP
│   ├── codex-cli/             # OpenAI Codex MCP
│   └── claude-code/           # Claude Code MCP
│
├── 03-security/               # Security & crypto
│   ├── COVERT_EDITION_SECURITY_ANALYSIS.md
│   ├── COVERT_EDITION_IMPLEMENTATION_CHECKLIST.md
│   ├── verify_covert_edition.sh
│   ├── procedures/            # Security procedures
│   └── audit/                 # Audit tools
│
├── 03-web-interface/          # Web UI
│   ├── clean_ui_v3.html       # Modern 3-panel interface
│   └── dsmil_unified_server.py # Backend with security
│
├── 04-hardware/               # Hardware integration
│   ├── ACCELERATOR_GUIDE.md
│   ├── NCS2_INTEGRATION.md
│   ├── NCS2_PERFORMANCE_OPTIMIZATION.md
│   ├── WHITERABBITNEO_GUIDE.md
│   ├── microcode/             # Intel microcode
│   └── ncs2-driver/           # NCS2 driver (submodule)
│
├── 04-integrations/           # RAG, web scraping, tools
│   ├── rag_manager.py         # 200+ doc knowledge base
│   ├── web_scraper.py         # Crawling + PDF extraction
│   ├── crawl4ai_wrapper.py    # crawl4ai integration
│   ├── autonomous_crawler.py  # Self-improving crawler
│   ├── rag_system/            # RAG components (200+ files)
│   │   ├── chunking.py        # Intelligent document chunking
│   │   ├── colbert_retrieval.py # ColBERT retrieval
│   │   ├── domain_finetuning.py # Domain-specific finetuning
│   │   ├── ACCURACY_OPTIMIZATION_ROADMAP.md
│   │   ├── PHASE1_OPTIMIZATIONS.md
│   │   ├── PHASE2_OPTIMIZATIONS.md
│   │   ├── PHASE3_OPTIMIZATIONS.md
│   │   └── benchmark_accuracy.py
│   ├── osint_comprehensive.py # 122 OSINT sources
│   ├── osint_uk_sources.py    # 45 UK-specific sources
│   └── (other RAG/OSINT components)
│
├── 05-deployment/             # Deployment configs
│   ├── npu-covert-edition.env # Environment variables
│   ├── verify_system.sh       # System verification
│   ├── systemd/               # systemd service files
│   └── zfs/                   # ZFS storage integration
│
├── 06-intel-systems/          # Intel/security systems
│   ├── INTEGRATION_GUIDE.md
│   ├── PRODUCTION_BEST_PRACTICES.md
│   ├── SCREENSHOT_INTEL_DEPLOYMENT.md
│   ├── deploy_screenshot_intel_production.sh
│   └── screenshot-analysis-system/
│
├── 99-archive/                # Legacy & historical
│   ├── database/              # Archive SQLite DB
│   ├── deployment-backup/     # Backup deployment scripts
│   └── (other legacy files)
│
├── ai_engine/                 # Python package root
├── screenshot_intel/          # Screenshot intelligence pkg
├── rag_system/                # RAG system package
├── avx512-unlock/             # AVX-512 unlock utility
├── packaging/                 # .deb package configs
│
├── .gitmodules                # Git submodule configs
├── setup.py                   # Python package setup
├── requirements.txt           # Dependencies (90+ packages)
├── __init__.py               # Package initialization
├── shrink_integration_manager.py # SHRINK manager
├── submodule_health_monitor.py  # Submodule health
│
├── setup-mcp-servers.sh       # MCP setup automation
├── launch-dsmil-control-center.sh # DSMIL control center
├── start-dashboard.sh         # Main platform launcher
│
├── README.md                  # Quick start guide
├── PROJECT_OVERVIEW.md        # Complete project overview
├── SUBMODULE_INTEGRATION.md   # Submodule integration guide
├── SHRINK_INTEGRATION.md      # SHRINK integration guide
├── DEPLOYMENT_READY.md        # Deployment documentation
├── PROJECT_100_PERCENT_COMPLETE.md
│
└── KP14/                      # Special projects directory
```

---

## 8. Database Architecture

### PostgreSQL Integration

**Schema Type:** Comprehensive relational with vector support

**Key Features:**
- **UUID-based identifiers** (all tables)
- **JSONB columns** for flexible metadata
- **Full-text search** support
- **Vector embeddings** (384-dimensional)
- **Temporal queries** (created_at, updated_at, accessed_at)
- **Hierarchical memory** (working/short-term/long-term tiers)

### Vector Database (Qdrant)

**Purpose:** Screenshot Intelligence RAG system

**Configuration:** 
- Host: 127.0.0.1 (default)
- Port: 6333
- Collections: Screenshot embeddings, document chunks
- Embedding Model: BAAI-large (88%+ accuracy)

**Integration:** `/home/user/LAT5150DRVMIL/04-integrations/rag_system/vector_rag_system.py`

---

## 9. Key Configuration Files

### MCP Server Configuration
- **File:** `02-ai-engine/mcp_servers_config.json`
- **Type:** JSON
- **Content:** 12 MCP servers with command, args, environment

### Database Schema
- **File:** `02-ai-engine/database_schema.sql`
- **Type:** PostgreSQL SQL
- **Tables:** 20+ tables, 34+ indexes, 6+ views, 4+ functions

### Model Configuration
- **File:** `02-ai-engine/models.json`
- **Type:** JSON
- **Content:** AI model definitions and parameters

### Hardware Profile
- **File:** `02-ai-engine/hardware_profile.json`
- **Type:** JSON
- **Content:** Dell Latitude 5450 specs, capabilities

### Environment Configuration
- **File:** `05-deployment/npu-covert-edition.env`
- **Type:** Shell environment variables
- **Content:** System settings, NPU configuration

### SHRINK Configuration
- **File:** `shrink_integration_manager.py`
- **Type:** Python dataclass configs
- **Content:** Compression settings, optimization parameters

### Security Configuration
- **File:** `02-ai-engine/mcp_security.py`
- **Type:** Python security manager
- **Features:** Token auth, rate limiting, audit logging

---

## 10. System Architecture Summary

### Three Core Subsystems

#### 1. DSMIL Hardware Layer
- **84 Devices** (656 operations)
- **7 Device Groups** (0x8000-0x806B)
- **79 Usable, 5 Quarantined** devices
- **SMI Interface** (ports 0xB2, 0xB3)
- **2GB Firmware Reserve**

#### 2. AI Intelligence Layer
- **5 Models:** fast (Phi-3), code (DeepSeek), quality (Llama), uncensored (WizardLM), large (Qwen)
- **Multi-Agent System:** 25+ specialized agents
- **RAG System:** 200+ documents, Qdrant vector DB
- **Post-Quantum Crypto:** CSNA 2.0 compliant
- **TPM 2.0:** 88 cryptographic algorithms

#### 3. Integration Layer
- **12 MCP Servers:** Including DIRECTEYE (pending), security tools, docs, OSINT
- **Screenshot Intelligence:** OCR + timeline + anomaly detection
- **OSINT:** 167+ sources (flights, ships, satellites, social media, news, etc.)
- **Web Integration:** crawl4ai, smart scrapers
- **SHRINK:** Storage optimization + compression

### Security Architecture

**Multi-Layer Security:**
1. **Token-based MCP authentication** (SHA-256)
2. **Rate limiting** (60 req/min per client)
3. **Input validation & sanitization**
4. **Path traversal prevention**
5. **Audit logging** (~/.dsmil/mcp_audit.log)
6. **Post-quantum cryptography**
7. **TPM 2.0 hardware attestation**

---

## 11. Integration Readiness Status

### Fully Integrated & Production Ready
- ✅ DSMIL subsystem (84 devices)
- ✅ AI engine (5 models)
- ✅ Screenshot intelligence
- ✅ RAG system (200+ docs)
- ✅ TPM 2.0 integration
- ✅ MCP servers (11/12 active)
- ✅ Web dashboard
- ✅ SHRINK compression

### Pending Integration
- ⏳ DIRECTEYE MCP (submodule configured but not initialized)
- ⏳ NCS2 driver (submodule configured but not initialized)

### Next Steps
```bash
# Initialize DIRECTEYE
git submodule update --init rag_system/mcp_servers/DIRECTEYE
cd rag_system/mcp_servers/DIRECTEYE
pip install -e .

# Update mcp_servers_config.json to include DIRECTEYE
# Add DIRECTEYE server entry
```

---

## 12. Performance & Optimization

### Included Optimization Documents
- `04-integrations/rag_system/PHASE1_OPTIMIZATIONS.md` - +11-20% accuracy gain
- `04-integrations/rag_system/PHASE2_OPTIMIZATIONS.md` - +10-17% additional gain
- `04-integrations/rag_system/PHASE3_OPTIMIZATIONS.md` - Advanced RAG optimizations
- `02-ai-engine/OPTIMIZATION_SUMMARY.md` - Complete optimization guide

### Hardware Optimizations
- **Meteor Lake Compiler Flags** - 15-30% faster compilation
- **AVX-512 Unlock** - 15-40% additional speedup (optional)
- **Intel NPU Support** - AI Boost VPU 3720 (48 TOPS)
- **Arc GPU Integration** - 28.6 TOPS GPU acceleration

---

## 13. Recommended Next Steps

### For DIRECTEYE Integration
1. Initialize the submodule: `git submodule update --init rag_system/mcp_servers/DIRECTEYE`
2. Install dependencies: `pip install -e rag_system/mcp_servers/DIRECTEYE`
3. Update MCP config: Add DIRECTEYE server to `mcp_servers_config.json`
4. Test: `python3 02-ai-engine/test_mcp_server.py`

### For Production Deployment
1. Review: `06-intel-systems/PRODUCTION_BEST_PRACTICES.md`
2. Deploy screenshot intelligence: `06-intel-systems/deploy_screenshot_intel_production.sh`
3. Verify system: `05-deployment/verify_system.sh`
4. Monitor health: `submodule_health_monitor.py`

### For Development
1. Install development dependencies: `pip install -e ".[screenshot_intel,api,dev]"`
2. Run tests: `04-integrations/rag_system/test_screenshot_intel_integration.py`
3. Validate system: `python3 02-ai-engine/system_validator.py --detailed`

---

## 14. Key Files Reference

| Purpose | File | Type |
|---------|------|------|
| Main Entry Point | `02-ai-engine/ai_gui_dashboard.py` | Python GUI |
| Web Dashboard | `03-web-interface/clean_ui_v3.html` + `dsmil_unified_server.py` | HTML + Python |
| AI Engine | `02-ai-engine/dsmil_ai_engine.py` | Python |
| MCP Config | `02-ai-engine/mcp_servers_config.json` | JSON |
| Database Schema | `02-ai-engine/database_schema.sql` | SQL |
| Security | `02-ai-engine/mcp_security.py` | Python |
| Hardware Control | `02-ai-engine/dsmil_subsystem_controller.py` | Python |
| Screenshot Intel | `screenshot_intel/screenshot_intelligence.py` | Python |
| RAG System | `04-integrations/rag_system/` | Python (200+ files) |
| SHRINK Manager | `shrink_integration_manager.py` | Python |
| Setup Scripts | `setup-mcp-servers.sh`, `start-dashboard.sh` | Bash |
| Documentation | `README.md`, `PROJECT_OVERVIEW.md`, `SUBMODULE_INTEGRATION.md` | Markdown |

---

## 15. Dependencies Summary

### Core AI Dependencies
- torch, transformers, datasets, accelerate
- intel-extension-for-pytorch, openvino
- sentence-transformers, qdrant-client
- langchain, llama-index

### Screenshot Intelligence
- paddleocr, paddlepaddle, pytesseract, Pillow
- watchdog, psutil

### MCP & Services
- mcp, fastapi, uvicorn, pydantic
- telethon (Telegram integration)

### Security & Crypto
- hashlib, os.urandom (built-in)
- Custom quantum crypto layer
- TPM 2.0 integration

### Development
- pytest, black, mypy, flake8

---

## Conclusion

LAT5150DRVMIL is a **comprehensive, production-ready AI platform** with:
- Deep hardware integration (84 DSMIL devices)
- Multi-model AI with intelligent routing
- Advanced security architecture
- Vector RAG system (Qdrant + BAAI)
- Screenshot intelligence with anomaly detection
- 167+ OSINT sources integrated
- Storage optimization (SHRINK)
- 12 MCP servers (11 active + 1 pending DIRECTEYE)

**DIRECTEYE integration** is configured but pending initialization. Once initialized, it will add 40+ blockchain intelligence services to the platform.

---

