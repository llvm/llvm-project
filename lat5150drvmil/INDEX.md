# LAT5150DRVMIL - Project Directory Index

> **Comprehensive directory structure and file location reference**
>
> Last updated: 2025-11-15

## 📁 Root Directory (Critical Files Only)

```
/
├── README.md                    # Main project documentation
├── INDEX.md                    # This file - comprehensive directory guide
├── dsmil_control_centre.py    # ⭐ DSMIL Control Centre (104-device management TUI)
├── dsmil.py                   # Core DSMIL module (imported by AI engine)
├── .gitignore                 # Git ignore patterns
├── .gitmodules                # Git submodule configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Python package setup
└── __init__.py               # Python package initialization
```

---

## 📚 Documentation (`00-documentation/`)

### Root Documentation (`00-documentation/00-root-docs/`)
*All project-level documentation moved here from root*

- **ALL_SUBMODULES.md** - Complete submodule inventory
- **BUILD_ON_HARDWARE.md** - Hardware build instructions
- **CODEBASE_EXPLORATION_REPORT.md** - Codebase analysis
- **CODEBASE_INTEGRATION_ANALYSIS.md** - Integration analysis
- **DEPLOYMENT_GUIDE.md** - Deployment procedures
- **DEPLOYMENT_READY.md** - Production readiness guide
- **DEPRECATION_PLAN.md** - Feature deprecation roadmap
- **DIRECTEYE_FULL_INTEGRATION.md** - DirectEye integration guide
- **INTEGRATED_SELF_AWARENESS.md** - Self-modification capabilities
- **NATURAL_LANGUAGE_INTEGRATION.md** - Natural language interface docs
- **PRODUCTION_READINESS_CHECKLIST.md** - Production checklist
- **PROJECT_100_PERCENT_COMPLETE.md** - Project completion status
- **PROJECT_NATURE.md** - Project philosophy and goals
- **PROJECT_OVERVIEW.md** - High-level project overview
- **QUICKSTART_NL_INTEGRATION.md** - Quick start for NL interface
- **QUICK_REFERENCE.md** - Quick reference guide
- **QUICK_START.md** - General quick start
- **REORGANIZATION_GUIDE.md** - Repository reorganization guide
- **SHRINK_INTEGRATION.md** - SHRINK submodule integration
- **START_HERE.md** - New user starting point
- **SUBMODULES_SUMMARY.md** - Submodule summary
- **SUBMODULE_INTEGRATION.md** - Submodule integration guide

### Other Documentation Directories
- **00-indexes/** - Index and organizational documents
- **01-planning/** - Planning and design documents
- **02-analysis/** - Analysis and research documents
- **03-ai-framework/** - AI framework documentation
- **03-operations/** - Operational procedures
- **04-progress/** - Progress tracking documents
- **05-reference/** - Reference materials
- **05-vulnerabilities/** - Security and vulnerability docs
- **06-tools/** - Tool-specific documentation

---

## 🔬 Source Code (`01-source/`)

### Core Components
- **kernel/** - Linux kernel modules and drivers
  - `core/` - DSMIL device drivers
    - `dsmil-84dev.c` - 84-device driver
    - `dsmil-104dev.c` - 104-device driver (latest)
    - `dsmil_driver_module.c` - Driver module entry point
  - `rust/` - Rust implementations
  - `README.md` - Driver documentation

- **serena-integration/** - Semantic code engine
  - `semantic_code_engine.py` - LSP-based code understanding

---

## 🤖 AI Engine (`02-ai-engine/`)

### Core AI Components
- **integrated_local_claude.py** - Main integrated AI system
- **natural_language_interface.py** - Conversational interface (with context optimization)
- **workflow_automation_hub.py** - Workflow automation system
- **unified_orchestrator.py** - Query routing and orchestration
- **advanced_planner.py** - Task planning engine
- **execution_engine.py** - Plan execution engine
- **ai.py** - AI model management
- **pattern_database.py** - Code pattern database
- **rag_system.py** - Retrieval-augmented generation

### Context Window Optimization (NEW)
- **advanced_context_optimizer.py** - Advanced context optimizer with 8 cutting-edge techniques
- **context_optimizer_integration.py** - Integration wrapper for AI engine
- **CONTEXT_OPTIMIZATION_PLAN.md** - Implementation plan and research foundation
- **CONTEXT_OPTIMIZATION_README.md** - User guide and API documentation
- **context_manager.py** - Basic context tracking (integrated)
- **ace_context_engine.py** - ACE-FCA patterns (integrated)
- **hierarchical_memory.py** - Three-tier memory (integrated)

**Features:**
- 40-60% optimal context window utilization
- Attention-based importance scoring
- Hierarchical summarization with content-specific strategies
- Semantic chunking with embeddings (Sentence-BERT)
- Dynamic pruning (5 strategies)
- Vector database retrieval (FAISS)
- Zero-loss compaction with three-tier memory
- Automatic integration with Natural Language Interface

### Specialized Components
- **shodan_search.py** - Shodan threat intelligence integration
- **codecraft_architect.py** - Production code templates
- **utilities/** - AI engine utilities

### Legacy Components
- **dsmil_ai_engine.py** - Original DSMIL AI engine
- **ai_query.py** - Query interface
- **ai_gui_dashboard.py** - GUI dashboard
- **ai_tui_v2.py** - Terminal UI

---

## 🛠️ Tools (`02-tools/`)

Various development and operational tools.

---

## 🔌 MCP Servers (`03-mcp-servers/`)

Model Context Protocol server implementations.

---

## 🔐 Security (`03-security/`)

Security tools, threat intelligence, and security frameworks.

---

## 🌐 Web Interface (`03-web-interface/`)

Web-based user interfaces and dashboards.

---

## ⚙️ Hardware (`04-hardware/`)

### Firmware (`04-hardware/firmware/`)
*Intel microcode and firmware files moved here*

- **Intel-Linux-Processor-Microcode-Data-Files-microcode-20240312.zip**
- **Intel-Linux-Processor-Microcode-Data-Files-microcode-20240813.zip**
- **i2p-archive-keyring.gpg** - I2P archive keyring

---

## 🔄 Integrations (`04-integrations/`)

### Integration Managers
- **shrink_integration_manager.py** - SHRINK integration management
- **shrink_intelligence_integration.py** - SHRINK intelligence integration

---

## 🚀 Deployment (`05-deployment/`)

Deployment configurations and tools.

---

## 💻 Intel Systems (`06-intel-systems/`)

Intel-specific system configurations and optimizations.

---

## 🪝 Hooks (`hooks/`)

### Crypto-POW (`hooks/crypto-pow/`)
- **crypto_pow.py** - Hardware-accelerated proof-of-work
- **README.md** - Crypto-POW documentation

### ShadowGit (`hooks/shadowgit/`)
- **shadowgit.py** - Git intelligence engine with NPU acceleration

---

## 📜 Scripts (`scripts/`)

### Installation & Setup
- **install.sh** - Main installation script
- **setup-mcp-servers.sh** - MCP server setup
- **migrate_to_v2.sh** - Version 2 migration

### Launch Scripts
- **launch-dsmil-control-center.sh** - DSMIL control center launcher
- **launch-ml-enhanced-activation.sh** - ML activation launcher
- **start-dashboard.sh** - Dashboard launcher

### Utilities
- **codex-guard.sh** - Code security guard
- **submodule_health_monitor.py** - Submodule health monitoring

---

## ⚙️ Configuration (`config/`)

### Configuration Files
- **codexrc.json** - Codex configuration
- **DSMIL_DEVICE_CAPABILITIES.json** - DSMIL device capability mapping

---

## 🧪 Tests (`tests/`)

### Test Files
- **test_shrink_integration.py** - SHRINK integration tests

---

## 📦 Archive (`99-archive/`, `_archived/`)

Deprecated and archived code (not actively maintained).

---

## 🗂️ Other Directories

- **KP14/** - KP14-specific components
- **ai_engine/** - Legacy AI engine code
- **avx512-unlock/** - AVX-512 unlocking utilities
- **deployment/** - Deployment-related files
- **packaging/** - Packaging and distribution
- **rag_system/** - RAG system implementation

---

## 🚀 Quick Start Guide

### For New Users
1. Start here: [`00-documentation/00-root-docs/START_HERE.md`](00-documentation/00-root-docs/START_HERE.md)
2. Read overview: [`00-documentation/00-root-docs/PROJECT_OVERVIEW.md`](00-documentation/00-root-docs/PROJECT_OVERVIEW.md)
3. Quick start: [`00-documentation/00-root-docs/QUICK_START.md`](00-documentation/00-root-docs/QUICK_START.md)

### For Development
1. Build guide: [`00-documentation/00-root-docs/BUILD_ON_HARDWARE.md`](00-documentation/00-root-docs/BUILD_ON_HARDWARE.md)
2. Integration guide: [`00-documentation/00-root-docs/SUBMODULE_INTEGRATION.md`](00-documentation/00-root-docs/SUBMODULE_INTEGRATION.md)
3. AI framework: [`00-documentation/03-ai-framework/`](00-documentation/03-ai-framework/)

### For Deployment
1. Deployment guide: [`00-documentation/00-root-docs/DEPLOYMENT_GUIDE.md`](00-documentation/00-root-docs/DEPLOYMENT_GUIDE.md)
2. Readiness checklist: [`00-documentation/00-root-docs/PRODUCTION_READINESS_CHECKLIST.md`](00-documentation/00-root-docs/PRODUCTION_READINESS_CHECKLIST.md)
3. Run installation: `./scripts/install.sh`
4. Launch dashboard: `./scripts/start-dashboard.sh`

---

## 🔍 Finding Files

### By Category

**Documentation**: `00-documentation/00-root-docs/*.md`
**AI Engine**: `02-ai-engine/*.py`
**Drivers**: `01-source/kernel/core/*.c`
**Scripts**: `scripts/*.sh`, `scripts/*.py`
**Configuration**: `config/*.json`
**Firmware**: `04-hardware/firmware/*`
**Integrations**: `04-integrations/*.py`
**Tests**: `tests/*.py`

### By Function

**Natural Language Interface**: `02-ai-engine/natural_language_interface.py`
**Context Window Optimization**: `02-ai-engine/advanced_context_optimizer.py`, `02-ai-engine/context_optimizer_integration.py`
**Workflow Automation**: `02-ai-engine/workflow_automation_hub.py`
**Search Integration**: `02-ai-engine/unified_orchestrator.py`, `02-ai-engine/shodan_search.py`
**Code Generation**: `02-ai-engine/codecraft_architect.py`
**Semantic Code Ops**: `01-source/serena-integration/semantic_code_engine.py`
**Git Intelligence**: `hooks/shadowgit/shadowgit.py`
**Secure Workflows**: `hooks/crypto-pow/crypto_pow.py`

---

## 📊 Directory Statistics

- **Total Directories**: 20+ top-level
- **Documentation Files**: 20+ markdown files
- **Shell Scripts**: 7 scripts
- **Python Modules**: 50+ AI engine modules
- **Firmware Files**: 3 firmware packages
- **Configuration Files**: 2 JSON configs

---

## 🔄 Recent Reorganization

**Date**: 2025-11-15

### What Changed
- ✅ Moved all documentation from root to `00-documentation/00-root-docs/`
- ✅ Moved all scripts from root to `scripts/`
- ✅ Moved firmware from root to `04-hardware/firmware/`
- ✅ Moved config files from root to `config/`
- ✅ Moved integration scripts to `04-integrations/`
- ✅ Moved tests to `tests/`
- ✅ Root directory now contains only critical files

### Path Updates
All documentation has been updated to reflect new paths. If you encounter broken links:
- Documentation: Prepend `00-documentation/00-root-docs/`
- Scripts: Prepend `scripts/`
- Config: Prepend `config/`

---

## 📝 Notes

- **Submodules**: Managed via `.gitmodules` - see `00-documentation/00-root-docs/ALL_SUBMODULES.md`
- **Dependencies**: Listed in `requirements.txt` at root
- **Setup**: Use `setup.py` for Python package installation
- **Entry Points**: Main scripts in `scripts/`, main module is `dsmil.py`

---

**For questions or issues, consult the documentation in `00-documentation/00-root-docs/` or check the comprehensive guides therein.**
