# LAT5150DRVMIL - Build Order & Installation Scripts

**Version**: 9.0.0
**Updated**: 2025-11-23
**Status**: Production Ready

This document enumerates all installation scripts and provides the canonical build order for integrating LAT5150DRVMIL as a submodule.

---

## Quick Reference

| Method | Script | Time | Use Case |
|--------|--------|------|----------|
| **Recommended** | `packaging/install-all-debs.sh` | 2 min | Production deployment |
| **Complete** | `scripts/install.sh` | 15 min | Full system with services |
| **Minimal** | `01-source/kernel/build/build-and-install.sh` | 3 min | Kernel driver only |
| **AI Stack** | `setup_complete_ai_stack.sh` | 20 min | AI models & dependencies |
| **MCP Servers** | `scripts/setup-mcp-servers.sh` | 10 min | External MCP servers |

---

## Installation Scripts Inventory

### Core Installation Scripts (5 Primary)

#### 1. **scripts/install.sh** - Complete System Installation
**Location**: `/scripts/install.sh`
**Purpose**: Full LAT5150 tactical AI sub-engine installation
**Duration**: 15-20 minutes
**Requires**: Root access, Internet connection

**What it installs**:
- System dependencies (Python 3.10+, gcc, make, Docker)
- Python virtual environment at `/opt/lat5150/venv`
- Ollama (local model runtime)
- Default AI models (whiterabbit-neo-33b, qwen2.5-coder, deepseek-r1)
- SystemD service (`lat5150-tactical-ai.service`)
- Master entrypoint (`/usr/local/bin/lat5150`)
- Self-awareness engine initialization
- Directory structure at `/opt/lat5150/`

**Usage**:
```bash
sudo ./scripts/install.sh
```

**Post-install**:
```bash
sudo systemctl start lat5150-tactical-ai
sudo systemctl enable lat5150-tactical-ai  # Auto-start on boot
http://localhost:5001  # Access UI
```

---

#### 2. **packaging/install-all-debs.sh** - DEB Package Installation (RECOMMENDED)
**Location**: `/packaging/install-all-debs.sh`
**Purpose**: Modular DEB package installation
**Duration**: 2-3 minutes
**Requires**: Root access, packages already built

**What it installs** (4 packages):
1. **dsmil-platform** (8.3.1-1) - 2.5 MB
   - Complete AI platform
   - ChatGPT-style interface
   - 7 auto-coding tools
   - RAG knowledge base
   - TPM attestation

2. **dell-milspec-tools** (1.0.0-1) - 24 KB
   - `dsmil-status` - Device status checker
   - `milspec-control` - MIL-SPEC feature control
   - `milspec-monitor` - System health monitoring
   - `tpm2-accel-status` - TPM2 acceleration status
   - `milspec-emergency-stop` - Emergency shutdown

3. **tpm2-accel-examples** (1.0.0-1) - 19 KB
   - SECRET level (security level 2) C examples
   - AES-256-GCM encryption demo
   - SHA3-512 hashing demo

4. **dsmil-complete** (8.3.2-1) - 1.5 KB
   - Meta-package (depends on all above)

**Usage**:
```bash
cd packaging
./build-all-debs.sh        # Build first (if not done)
sudo ./install-all-debs.sh # Install in dependency order
./verify-installation.sh   # 10-point verification
```

**Installed locations**:
- Binaries: `/usr/local/bin/`
- Libraries: `/usr/local/lib/lat5150/`
- Config: `/etc/lat5150/`
- Docs: `/usr/local/share/doc/lat5150/`

---

#### 3. **01-source/kernel/build/build-and-install.sh** - Kernel Driver Build
**Location**: `/01-source/kernel/build/build-and-install.sh`
**Purpose**: Build and install DSMIL kernel drivers
**Duration**: 3-5 minutes
**Requires**: Root access, kernel headers, Rust toolchain (auto-installs if missing)

**What it builds**:
- **dsmil-84dev.ko** - 84-device driver with Rust safety layer
- **dsmil-72dev.ko** - Legacy 72-device compatibility alias

**Features**:
- Auto-detects Rust toolchain (rustc, cargo)
- Auto-installs rust-src component (prevents FMA instruction issues)
- Falls back to C stubs if Rust unavailable
- Handles objtool errors gracefully
- Unloads existing drivers before rebuild
- Creates compatibility aliases for legacy tooling

**Usage**:
```bash
cd 01-source/kernel/build
sudo ./build-and-install.sh
```

**Verification**:
```bash
lsmod | grep dsmil          # Check loaded
ls -la /dev/dsmil*          # Check device nodes
dmesg | grep -i dsmil       # Check kernel log
```

---

#### 4. **setup_complete_ai_stack.sh** - AI Stack Installation
**Location**: `/setup_complete_ai_stack.sh` (root)
**Purpose**: Install all AI dependencies and models
**Duration**: 20-30 minutes (includes model downloads)
**Requires**: Internet connection, Python 3.9+

**What it installs**:

**Mandatory AI Dependencies**:
- `openai>=1.0.0` - OpenAI API (structured outputs)
- `google-generativeai>=0.8.0` - Gemini API (multimodal)
- `pydantic>=2.9.0` - Type safety
- `pydantic-ai>=0.0.13` - AI-native type-safe framework
- `ollama>=0.3.0` - Local model client
- `duckduckgo-search>=6.3.0` - Privacy-first web search
- `shodan>=1.31.0` - Threat intelligence
- `httpx>=0.27.0` - Async HTTP
- `fastapi>=0.115.0` - API framework
- `uvicorn[standard]>=0.30.0` - ASGI server

**Heavy ML Dependencies** (optional, skip with `--skip-heavy`):
- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - Hugging Face transformers
- `sentence-transformers>=2.2.0` - Embeddings
- `intel-extension-for-pytorch>=2.0.0` - Intel optimizations
- `numpy`, `scipy`, `pandas` - Scientific computing

**Ollama Models**:
- `whiterabbit-neo-33b` - Primary model (NPU/GPU/NCS2 accelerated)
- `qwen2.5-coder:7b` - Quality code generation
- `deepseek-r1:1.5b` - Legacy fast model

**Usage**:
```bash
./setup_complete_ai_stack.sh                 # Full install
./setup_complete_ai_stack.sh --skip-heavy    # Skip PyTorch/Transformers
```

**API Keys** (optional):
```bash
export OPENAI_API_KEY='sk-...'      # OpenAI
export GEMINI_API_KEY='AI...'       # Google Gemini
export SHODAN_API_KEY='...'         # Shodan
```

---

#### 5. **scripts/setup-mcp-servers.sh** - MCP Servers Installation
**Location**: `/scripts/setup-mcp-servers.sh`
**Purpose**: Install external MCP servers
**Duration**: 10-15 minutes
**Requires**: Node.js 18+, npm, Python 3.10+

**What it installs** (5 external + 6 core):

**External MCP Servers**:
1. **search-tools-mcp** - Advanced code search (ripgrep, ast-grep)
2. **docs-mcp-server** - Documentation indexing
3. **MetasploitMCP** - Security testing framework
4. **mcp-maigret** - Username OSINT
5. **mcp-for-security** - 23 security tools (Nmap, Nuclei, SQLmap, etc.)

**Core Python Servers** (already included):
1. **dsmil-ai** - DSMIL AI engine with RAG
2. **sequential-thinking** - Multi-step reasoning
3. **filesystem** - Sandboxed file operations
4. **memory** - Persistent knowledge graph
5. **fetch** - Web content fetching
6. **git** - Git operations

**Usage**:
```bash
./scripts/setup-mcp-servers.sh
```

**Configuration**: `02-ai-engine/mcp_servers_config.json`

---

### Secondary Installation Scripts (Component-Specific)

#### AI Engine Setup
- **`02-ai-engine/install_mcp.sh`** - MCP protocol installation
- **`02-ai-engine/install_pydantic_ai.sh`** - Pydantic AI framework
- **`02-ai-engine/setup_ai_enhancements.sh`** - AI enhancements
- **`02-ai-engine/setup_uncensored_models.sh`** - Uncensored models (WizardLM)
- **`02-ai-engine/setup_unified_ai_platform.sh`** - Unified platform

#### TPM/Hardware Integration
- **`02-ai-engine/tpm2_compat/install_native_integration.sh`** - TPM2 native integration
- **`02-ai-engine/tpm2_compat/c_acceleration/install_tpm2_module.sh`** - TPM2 C acceleration module
- **`02-ai-engine/tpm2_compat/c_acceleration/uninstall_tpm2_module.sh`** - Uninstall TPM2 module

#### MCP & Integrations
- **`03-mcp-servers/setup_mcp_servers.sh`** - Main MCP setup (called by scripts/setup-mcp-servers.sh)
- **`03-mcp-servers/setup_mcp_docker.sh`** - Dockerized MCP servers
- **`04-integrations/install_crawl4ai.sh`** - Industrial web crawler
- **`04-integrations/rag_system/install_cve_service.sh`** - CVE scraper service
- **`04-integrations/rag_system/uninstall_cve_service.sh`** - Uninstall CVE scraper

#### RAG & Knowledge Base
- **`04-integrations/rag_system/setup_code_assistant.sh`** - Code assistant
- **`04-integrations/rag_system/setup_peft.sh`** - PEFT (Parameter-Efficient Fine-Tuning)
- **`04-integrations/rag_system/setup_transformer.sh`** - Transformer models

#### Forensics & Security
- **`04-integrations/forensics/setup_tools.sh`** - Forensics tools

#### System Integration
- **`deployment/install-autostart.sh`** - Auto-start on boot
- **`deployment/install-self-improvement-timer.sh`** - Self-improvement timer
- **`deployment/install-unified-api-autostart.sh`** - Unified API auto-start
- **`deployment/setup-shell-integration.sh`** - Shell integration

#### Hardware Specific
- **`scripts/install-ncs2.sh`** - Intel Neural Compute Stick 2
- **`06-intel-systems/screenshot-analysis-system/setup_screenshot_intel.sh`** - Screenshot intelligence

#### File Manager Integration
- **`02-ai-engine/file_manager_integration/install_context_menu.sh`** - Context menu integration

---

## Recommended Build Order for Submodule Integration

### Method 1: Quick Start (DEB Packages) - 5 Minutes

**Best for**: Production deployment, CI/CD, Docker containers

```bash
# 1. Add as submodule
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git external/lat5150
git submodule update --init --recursive

# 2. Build DEB packages
cd external/lat5150/packaging
./build-all-debs.sh

# 3. Install packages
sudo ./install-all-debs.sh

# 4. Verify
./verify-installation.sh
```

**Result**: All binaries, libraries, and tools installed system-wide

---

### Method 2: Complete System Setup - 30 Minutes

**Best for**: Development workstation, full-featured deployment

```bash
# 1. Add as submodule
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git external/lat5150
cd external/lat5150

# 2. Install AI stack (mandatory)
./setup_complete_ai_stack.sh

# 3. Install system (SystemD service, directories, models)
sudo ./scripts/install.sh

# 4. Install MCP servers
./scripts/setup-mcp-servers.sh

# 5. Build kernel driver (if on Dell MIL-SPEC hardware)
cd 01-source/kernel/build
sudo ./build-and-install.sh

# 6. Start services
sudo systemctl start lat5150-tactical-ai
sudo systemctl enable lat5150-tactical-ai

# 7. Access UI
# Open http://localhost:5001
```

**Result**: Full tactical AI platform with all features

---

### Method 3: Minimal Kernel Driver Only - 3 Minutes

**Best for**: Hardware access only, embedded systems

```bash
# 1. Add as submodule
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git external/lat5150
cd external/lat5150

# 2. Build and install kernel driver
cd 01-source/kernel/build
sudo ./build-and-install.sh

# 3. Verify
lsmod | grep dsmil
ls -la /dev/dsmil*
```

**Result**: DSMIL kernel driver loaded, 84 devices accessible

---

### Method 4: Python-Only Integration - 10 Minutes

**Best for**: Python projects, API access, no system changes

```bash
# 1. Add as submodule
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git external/lat5150
cd external/lat5150

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install AI stack
./setup_complete_ai_stack.sh --skip-heavy

# 4. Use in your code
```

**Python usage**:
```python
import sys
sys.path.insert(0, 'external/lat5150/02-ai-engine')

from dsmil_subsystem_controller import DSMILController
from quantum_crypto_layer import QuantumCryptoLayer

controller = DSMILController()
devices = controller.list_safe_devices()  # 99 safe devices
```

**Result**: Python API access, no system installation

---

## Build Order Dependencies

### Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│ 1. System Dependencies                              │
│    - Python 3.10+, gcc, make, kernel headers        │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌────────────────────┐  ┌─────────────────────┐
│ 2. AI Stack        │  │ 2. Kernel Driver    │
│    (setup_ai.sh)   │  │    (build-and-      │
│                    │  │     install.sh)     │
└─────────┬──────────┘  └──────────┬──────────┘
          │                        │
          ▼                        ▼
┌─────────────────────────────────────────────────────┐
│ 3. Core Installation                                 │
│    - scripts/install.sh (SystemD, directories)       │
│    OR                                                │
│    - packaging/install-all-debs.sh (DEB packages)   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ 4. Optional Components                               │
│    - MCP Servers (setup-mcp-servers.sh)             │
│    - TPM Integration (install_native_integration.sh)│
│    - RAG System (setup_peft.sh)                     │
│    - Forensics (setup_tools.sh)                     │
└─────────────────────────────────────────────────────┘
```

### Critical Dependencies

| Component | Requires | Must Run Before |
|-----------|----------|-----------------|
| **Kernel Driver** | kernel headers, gcc, Rust | Any driver usage |
| **AI Stack** | Python 3.10+, pip | Core installation |
| **Core Installation** | AI stack OR minimal deps | Service start |
| **MCP Servers** | Node.js 18+, npm | MCP usage |
| **DEB Packages** | dpkg, apt | Package usage |
| **TPM Integration** | tpm2-tools, libtpm2-dev | Hardware attestation |

---

## Verification Commands

### After Kernel Driver Installation
```bash
# Check module loaded
lsmod | grep dsmil

# Check device nodes
ls -la /dev/dsmil*

# Check kernel messages
dmesg | grep -i dsmil | tail -20

# Test device access
python3 -c "
from dsmil_subsystem_controller import DSMILController
c = DSMILController()
print(f'Safe devices: {len(c.list_safe_devices())}')
"
```

### After DEB Package Installation
```bash
# Check installed packages
dpkg -l | grep -E "dsmil|milspec"

# Check binaries
which dsmil-status milspec-control

# Run status check
dsmil-status

# Verify installation
cd packaging && ./verify-installation.sh
```

### After Complete Installation
```bash
# Check SystemD service
sudo systemctl status lat5150-tactical-ai

# Check logs
sudo journalctl -u lat5150-tactical-ai -n 50

# Check UI
curl -s http://localhost:5001/api/v2/self-awareness | jq '.system_name'

# Check Ollama models
ollama list

# Check MCP servers
cat 02-ai-engine/mcp_servers_config.json | jq '.mcpServers | keys'
```

---

## Troubleshooting Build Order Issues

### Issue: Kernel headers not found
```bash
# Solution: Install kernel headers first
sudo apt-get install linux-headers-$(uname -r)
```

### Issue: Rust toolchain not found
```bash
# Solution: Install Rust (build-and-install.sh auto-installs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup component add rust-src
```

### Issue: Python version too old
```bash
# Check version
python3 --version

# Solution: Use Python 3.10+ or deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.10 python3.10-venv python3.10-dev
```

### Issue: Node.js too old for MCP servers
```bash
# Check version
node --version

# Solution: Install Node 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Issue: DEB package conflicts
```bash
# Solution: Uninstall old packages first
sudo apt-get remove dsmil-platform dell-milspec-tools
sudo apt-get autoremove
# Then reinstall
cd packaging && sudo ./install-all-debs.sh
```

### Issue: Driver already registered
```bash
# Solution: Unload all DSMIL drivers
sudo rmmod dsmil_84dev dsmil_72dev 2>/dev/null
lsmod | grep dsmil  # Should show nothing
# Then rebuild
cd 01-source/kernel/build && sudo ./build-and-install.sh
```

---

## Integration Patterns for Parent Projects

### Makefile Integration
```makefile
LAT5150_DIR := external/lat5150

.PHONY: lat5150-debs lat5150-kernel lat5150-ai lat5150-all

lat5150-debs:
	cd $(LAT5150_DIR)/packaging && \
	./build-all-debs.sh && \
	sudo ./install-all-debs.sh

lat5150-kernel:
	cd $(LAT5150_DIR)/01-source/kernel/build && \
	sudo ./build-and-install.sh

lat5150-ai:
	cd $(LAT5150_DIR) && \
	./setup_complete_ai_stack.sh && \
	./scripts/setup-mcp-servers.sh

lat5150-all: lat5150-ai lat5150-kernel lat5150-debs
	@echo "LAT5150 integration complete"
```

### CMake Integration
```cmake
# Add LAT5150 submodule
add_subdirectory(external/lat5150 EXCLUDE_FROM_ALL)

# Custom targets
add_custom_target(lat5150-debs
    COMMAND bash -c "cd packaging && ./build-all-debs.sh && sudo ./install-all-debs.sh"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/external/lat5150
    COMMENT "Building LAT5150 DEB packages"
)

add_custom_target(lat5150-kernel
    COMMAND sudo bash build-and-install.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/external/lat5150/01-source/kernel/build
    COMMENT "Building LAT5150 kernel driver"
)
```

### Docker Integration
```dockerfile
FROM ubuntu:22.04

# Copy submodule
COPY external/lat5150 /opt/lat5150

# Install dependencies and AI stack
RUN cd /opt/lat5150 && \
    ./setup_complete_ai_stack.sh && \
    ./scripts/setup-mcp-servers.sh

# Install DEB packages
RUN cd /opt/lat5150/packaging && \
    ./build-all-debs.sh && \
    ./install-all-debs.sh

# Kernel driver (skip in Docker, needs host kernel)
# RUN cd /opt/lat5150/01-source/kernel/build && ./build-and-install.sh

EXPOSE 5001 5050
CMD ["/usr/local/bin/lat5150"]
```

---

## Time Estimates

| Installation Method | Duration | Components |
|---------------------|----------|------------|
| **Minimal** (kernel only) | 3 min | Driver |
| **Quick** (DEB packages) | 5 min | Packages, tools |
| **Standard** (AI + DEB) | 15 min | AI stack, packages |
| **Complete** (everything) | 30 min | AI, MCP, driver, services |
| **Full Custom** (all options) | 45 min | Everything + extras |

**Network-dependent**: Model downloads (whiterabbit-neo-33b ~18GB) add 10-60 minutes depending on connection speed.

---

## Summary

**For submodule integration, the recommended build order is**:

1. **Add submodule**: `git submodule add`
2. **Build DEB packages**: `packaging/build-all-debs.sh`
3. **Install packages**: `sudo packaging/install-all-debs.sh`
4. **Verify**: `packaging/verify-installation.sh`

This provides the fastest, cleanest integration with all tools available system-wide.

For development or custom integration, use **Method 2 (Complete System Setup)** for full functionality.

For minimal hardware-only access, use **Method 3 (Kernel Driver Only)**.

---

**See also**:
- [SUBMODULE_INTEGRATION.md](SUBMODULE_INTEGRATION.md) - Complete integration guide
- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start
- [packaging/BUILD_INSTRUCTIONS.md](packaging/BUILD_INSTRUCTIONS.md) - DEB package build guide

---

**LAT5150DRVMIL v9.0.0** | Build Order Reference | Updated 2025-11-23
