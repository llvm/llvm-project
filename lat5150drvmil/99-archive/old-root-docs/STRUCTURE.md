# DSMIL Unified AI Platform - Directory Structure

## Core Directories

### `00-documentation/`
Complete documentation for the DSMIL platform
- `00-indexes/` - Index files and catalogs
- `01-planning/` - Project planning documents
- `02-analysis/` - Technical analysis
- `03-ai-framework/` - AI framework documentation
- `04-progress/` - Progress reports and changelogs
- `05-reference/` - Reference materials
- `00-root-docs/` - Miscellaneous root-level docs
- `archive/` - Historical documentation

### `01-source/`
Original DSMIL framework source code
- `kernel/` - Kernel module source
- `kernel-driver/` - Kernel driver
- `userspace-tools/` - Userspace utilities
- `systemd/` - Systemd integration
- `tests/` - Test suites
- `security_chaos_framework/` - Security testing framework

### `02-ai-engine/`
AI inference engine and model management
- `dsmil_ai_engine.py` - Main AI engine
- `smart_router.py` - Smart model routing
- `code_specialist.py` - Code generation specialist
- `local_claude_code.py` - Local code editing
- `web_search.py` - Web search integration
- `unified_orchestrator.py` - Multi-backend orchestration
- `sub_agents/` - Specialized sub-agents

### `03-web-interface/`
Web-based user interface
- `clean_ui_v3.html` - Modern ChatGPT-style UI
- `dsmil_unified_server.py` - Backend server
- `military_terminal_v2.html` - Alternative terminal UI
- Documentation for RAG and web features

### `03-security/`
Security documentation and procedures
- `audit/` - Security audit reports
- `procedures/` - Safety and security procedures
- Covert Edition documentation

### `04-integrations/`
External integrations and tools
- `rag_manager.py` - RAG knowledge base
- `web_scraper.py` - Intelligent web crawler
- `crawl4ai_wrapper.py` - Industrial crawler integration

### `05-deployment/`
Deployment configuration and scripts
- `systemd/` - Systemd service files
- `npu-covert-edition.env` - Covert Edition environment
- `verify_system.sh` - System verification

### `99-archive/`
Archived code, old versions, and backups
- Historical versions
- Cleanup backups
- Deprecated code

## Build and Packaging

### `build/`
Build artifacts and compiled binaries

### `packaging/`
Debian packages and distribution files
- `debian/` - Debian packaging
- `dkms/` - DKMS module packaging

### `tpm2_compat/`
TPM 2.0 compatibility layer
- `core/` - Core TPM functionality
- `tools/` - TPM utilities

## Root Files

### Installation
- `install.sh` - Automated installer
- `uninstall.sh` - Uninstaller
- `INSTALL.md` - Installation guide

### Documentation
- `README.md` - Main documentation
- `STRUCTURE.md` - This file

### Configuration
- `.gitignore` - Git ignore patterns
- `DSMIL_UNIVERSAL_FRAMEWORK.py` - Universal framework

### Logs
- `logs/` - Application logs
- `health_log.jsonl` - Health monitoring logs

---

**Last Updated:** $(date +%Y-%m-%d)
**Version:** 8.3
