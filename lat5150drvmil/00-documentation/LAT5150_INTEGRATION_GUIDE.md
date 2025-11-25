# LAT5150 DRVMIL - Complete Integration Guide

**Version**: 2.0.0
**Last Updated**: 2025-11-17
**Status**: Production Ready

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Submodule Directory Structure](#submodule-directory-structure)
4. [Build Order & Dependencies](#build-order--dependencies)
5. [Component Integration](#component-integration)
6. [Entry Points](#entry-points)
7. [API Integration](#api-integration)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

LAT5150 DRVMIL is a comprehensive AI-driven military intelligence and tactical operations system featuring:

- **Advanced AI Engine** with self-improvement capabilities
- **Red Team Benchmark** for offensive security testing
- **Heretic Abliteration** for refusal removal and model optimization
- **Atomic Red Team** MITRE ATT&CK integration
- **Hardware Access** via DSMIL driver interface
- **MCP Server Integration** for extensible capabilities
- **Natural Language Interface** on port 80/5001
- **Web-Based Dashboard** with full system control

---

## Architecture

```
LAT5150 DRVMIL
│
├── 00-documentation/          Documentation and guides
├── 01-source/                 Core source code (C/C++/Rust)
├── 02-ai-engine/              AI processing and model management
│   ├── enhanced_ai_engine.py         Main AI engine
│   ├── redteam_ai_benchmark.py       12 offensive security tests
│   ├── ai_self_improvement.py        Automated improvement loop
│   ├── heretic_abliteration.py       Refusal removal system
│   ├── atomic_red_team_api.py        MITRE ATT&CK integration
│   └── ...                           Other AI components
│
├── 03-mcp-servers/            Model Context Protocol servers
│   ├── atomic-red-team-data/         Atomic Red Team test data
│   └── ...                           Other MCP servers
│
├── 03-web-interface/          Web UI and APIs
│   ├── unified_tactical_api.py       Natural language API (port 80)
│   ├── capability_registry.py        24 registered capabilities
│   ├── natural_language_processor.py NL query processing
│   └── ...                           Dashboard and interfaces
│
├── 04-hardware/               Hardware interface layer
├── 05-deployment/             Deployment scripts and configs
│   ├── systemd/                      SystemD service files
│   ├── install-unified-api-autostart.sh
│   ├── install-self-improvement-timer.sh
│   ├── lat5150-api-env.sh           Shell helper functions
│   └── ...                           Deployment tools
│
└── lat5150.sh                 **ROOT ENTRY POINT**
```

---

## Submodule Directory Structure

### **00-documentation/**
- System architecture documentation
- API references
- Setup guides
- Technical reports

**Key Files**:
- `SYSTEM_ARCHITECTURE.md` - Overall system design
- `API_REFERENCE.md` - Complete API documentation
- Various setup and integration guides

### **01-source/**
Core C/C++/Rust implementations:
- DSMIL driver interface
- Hardware abstraction layers
- Performance-critical components

**Build Requirements**:
- GCC/Clang for C/C++
- Rust toolchain for Rust components
- Linux kernel headers

### **02-ai-engine/**
Python-based AI processing core:

**Core Components**:
- `enhanced_ai_engine.py` - Main AI engine with model management
- `model_manager.py` - Model loading and inference
- `context_manager.py` - Context window optimization

**Red Team & Security**:
- `redteam_ai_benchmark.py` - 12 offensive security tests
- `ai_self_improvement.py` - Automated improvement loop
- `heretic_abliteration.py` - Refusal removal via orthogonal projection
- `heretic_web_api.py` - REST API for Heretic operations
- `atomic_red_team_api.py` - MITRE ATT&CK test integration

**Integrations**:
- `dsmil_integration_adapter.py` - Hardware interface
- `ai_system_integrator.py` - Component orchestration
- `agent_orchestrator.py` - Multi-agent coordination

**Build Dependencies**:
```bash
pip install -r requirements.txt
# Key packages:
# - torch, transformers (AI models)
# - flask, flask-cors (APIs)
# - optuna (Heretic optimization)
# - pydantic-settings (configuration)
```

### **03-mcp-servers/**
Model Context Protocol server implementations:

**atomic-red-team-data/**:
- MITRE ATT&CK technique definitions
- Test case YAML files
- Execution logs and results

**Integration**:
- Uses `uvx atomic-red-team-mcp` package
- Configured via environment variables
- Accessed via stdio transport

### **03-web-interface/**
Web UI and unified APIs:

**Core APIs**:
- `unified_tactical_api.py` - Natural language interface (port 80)
- `capability_registry.py` - 24 capabilities registered
- `natural_language_processor.py` - NL query parsing

**Dashboards**:
- `ai_gui_dashboard.py` - Main dashboard (port 5050)
- TEMPEST integration - System control panel

**Build Dependencies**:
```bash
pip install flask flask-cors
```

### **04-hardware/**
Hardware interface layers:
- DSMIL driver bindings
- TPM2 integration
- Device discovery and enumeration

### **05-deployment/**
Production deployment configurations:

**SystemD Services**:
- `lat5150-unified-api.service` - Unified API (port 80)
- `lat5150-self-improvement.service` - Self-improvement runs
- `lat5150-self-improvement.timer` - Daily schedule (2 AM)

**Installation Scripts**:
- `install-unified-api-autostart.sh` - One-command API setup
- `install-self-improvement-timer.sh` - Scheduled improvement setup
- `setup-shell-integration.sh` - Add helpers to .bashrc

**Documentation**:
- `AUTO_INSTALL_README.md` - Unified API installation
- `AUTO_IMPROVEMENT_README.md` - Self-improvement system guide

---

## Build Order & Dependencies

### Phase 1: System Prerequisites

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.10+
sudo apt install python3 python3-pip python3-venv -y

# 3. Install Rust (for MCP servers)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 4. Install uv/uvx (for atomic-red-team-mcp)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Phase 2: Python Dependencies

```bash
cd /home/user/LAT5150DRVMIL

# Install core AI engine dependencies
pip3 install flask flask-cors torch transformers optuna pydantic-settings

# Optional: Install full requirements (if requirements.txt exists)
# pip3 install -r 02-ai-engine/requirements.txt
```

### Phase 3: Build C/C++ Components (if needed)

```bash
# Build DSMIL components
cd 01-source/dsmil
make clean && make

# Build other native components as needed
```

### Phase 4: Deploy Services

```bash
cd /home/user/LAT5150DRVMIL/deployment

# Deploy unified tactical API (port 80)
sudo ./install-unified-api-autostart.sh install

# Deploy self-improvement timer (daily at 2 AM)
sudo ./install-self-improvement-timer.sh install

# Add shell helpers to .bashrc
./setup-shell-integration.sh
source ~/.bashrc
```

### Phase 5: Verification

```bash
# Check services
sudo systemctl status lat5150-unified-api
sudo systemctl status lat5150-self-improvement.timer

# Test unified API
curl http://localhost:80/api/self-awareness | jq '.'

# Test natural language query
lat5150_query "show system status"

# Verify shell functions
lat5150_test
```

---

## Component Integration

### 1. Enhanced AI Engine Integration

**File**: `02-ai-engine/enhanced_ai_engine.py`

**Key Methods**:
```python
from enhanced_ai_engine import EnhancedAIEngine

# Initialize
engine = EnhancedAIEngine(
    user_id="system",
    enable_self_improvement=True
)

# Query
response = engine.query("Your prompt here", temperature=0.7)

# Abliterate model
result = engine.abliterate_model(
    model_name="uncensored_code",
    optimization_trials=200,
    save_results=True
)
```

**Integration Points**:
- Heretic abliteration via `abliterate_model()`
- Model management via `model_manager`
- Context optimization via `context_manager`
- Hardware access via `dsmil_integration_adapter`

### 2. Red Team Benchmark Integration

**File**: `02-ai-engine/redteam_ai_benchmark.py`

**Usage**:
```python
from redteam_ai_benchmark import RedTeamBenchmark
from enhanced_ai_engine import EnhancedAIEngine

# Initialize
benchmark = RedTeamBenchmark(model_name="uncensored_code")
engine = EnhancedAIEngine(user_id="benchmark")

# Run benchmark (12 offensive security tests)
summary = benchmark.run_benchmark(engine=engine)

print(f"Score: {summary.percentage}%")
print(f"Verdict: {summary.verdict}")
print(f"Refusals: {summary.refused_count}/12")
print(f"Correct: {summary.correct_count}/12")
```

**Test Categories**:
1. AMSI Bypass (T1562.006)
2. ADCS ESC1 exploitation
3. NTLM Relay to ADCS (ESC8)
4. Manual PE mapping injection
5. ETW bypass techniques
6. UAC bypass via registry
7. C2 profile emulation
8. Direct syscall shellcode
9. ADCS ESC12 attacks
10. LDAP relay with RBCD
11. EDR unhooking methods
12. Social engineering payloads

### 3. Self-Improvement Loop Integration

**File**: `02-ai-engine/ai_self_improvement.py`

**Usage**:
```python
from ai_self_improvement import AISelfImprovement

# Initialize
improver = AISelfImprovement(
    model_name="uncensored_code",
    target_score=80.0,
    max_cycles=5
)

# Run full improvement session
session = improver.run_full_improvement_session()

# Results
print(f"Initial: {session.initial_score}%")
print(f"Final: {session.final_score}%")
print(f"Improvement: +{session.total_improvement}%")
print(f"Target met: {session.target_reached}")
```

**Workflow**:
```
1. Run benchmark → Get score (e.g., 58%)
2. Analyze results → Detect 3 refusals, 2 hallucinations
3. Apply Heretic abliteration → Remove refusals
4. Re-run benchmark → New score (e.g., 75%)
5. Repeat until ≥80% or plateau
```

### 4. Heretic Abliteration Integration

**File**: `02-ai-engine/heretic_abliteration.py`

**Direct Usage**:
```python
from heretic_abliteration import HereticModelWrapper
import torch

# Load model
model = ...  # Your transformer model
tokenizer = ...  # Your tokenizer

# Initialize wrapper
wrapper = HereticModelWrapper(model, tokenizer, device="cuda")

# Define prompts
good_prompts = ["Explain photosynthesis", ...]
bad_prompts = ["How to bypass security", ...]

# Define parameters
parameters = {
    "mlp": AbliterationParameters(
        max_weight=1.0,
        min_weight=0.5,
        min_weight_distance=0.3
    )
}

# Run abliteration
refusal_directions = wrapper.full_abliteration_workflow(
    good_prompts=good_prompts,
    bad_prompts=bad_prompts,
    parameters=parameters
)

# Model is now abliterated in-place
```

**Via EnhancedAIEngine** (recommended):
```python
from enhanced_ai_engine import EnhancedAIEngine

engine = EnhancedAIEngine()
result = engine.abliterate_model("uncensored_code", trials=200)
```

### 5. Atomic Red Team Integration

**File**: `02-ai-engine/atomic_red_team_api.py`

**Usage**:
```python
from atomic_red_team_api import AtomicRedTeamAPI

# Initialize
art = AtomicRedTeamAPI()

# Query techniques
tests = art.query_tests("T1059.002")

# Search by platform
results = art.search_atomics("Windows", "powershell")

# List all techniques
all_techniques = art.list_all_techniques()

# Refresh from GitHub
art.refresh_tests()
```

**Via Natural Language** (recommended):
```bash
curl -X POST http://localhost/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me atomic tests for T1059.002"}'
```

### 6. Unified Tactical API Integration

**File**: `03-web-interface/unified_tactical_api.py`

**Natural Language Queries**:
```bash
# System status
curl -X POST http://localhost/api/query \
  -d '{"query": "check system health"}'

# Red team benchmark
curl -X POST http://localhost/api/query \
  -d '{"query": "run red team benchmark"}'

# Self-improvement
curl -X POST http://localhost/api/query \
  -d '{"query": "improve yourself"}'

# Atomic Red Team
curl -X POST http://localhost/api/query \
  -d '{"query": "list MITRE ATT&CK techniques"}'

# Hardware access
curl -X POST http://localhost/api/query \
  -d '{"query": "scan hardware devices"}'
```

**Registered Capabilities** (24 total):

**Atomic Red Team (4)**:
- `atomic_query_tests` - Query test cases
- `atomic_list_techniques` - List MITRE techniques
- `atomic_refresh` - Update from GitHub
- `atomic_validate` - Validate YAML structure

**Red Team Benchmark (4)**:
- `redteam_run_benchmark` - Run 12 security tests
- `redteam_get_results` - Get latest results
- `self_improve` - Run improvement session
- `self_improve_status` - Get improvement status

**System Capabilities (16)**:
- Code understanding (Serena LSP)
- Hardware access (DSMIL)
- Agent execution (AgentSystems)
- Security & audit
- System control & monitoring

---

## Entry Points

### 1. Root Launcher Script

**File**: `/home/user/LAT5150DRVMIL/lat5150.sh`

```bash
./lat5150.sh [command]

Commands:
  start-all       Start all services
  stop-all        Stop all services
  status          Show system status
  dashboard       Launch web dashboard
  benchmark       Run red team benchmark
  improve         Run self-improvement
  test            Run integration tests
  logs            Follow service logs
```

### 2. Shell Helper Functions

**File**: `deployment/lat5150-api-env.sh`

Sourced in `.bashrc`, provides:
```bash
lat5150_query <query>           # Natural language query
lat5150_status                  # System status
lat5150_atomic_list <technique> # Atomic tests
lat5150_atomic_search <platform> # Search tests
lat5150_capabilities            # Show all 24 capabilities
lat5150_test                    # Run health tests
lat5150_logs                    # View service logs
```

### 3. SystemD Services

**Unified API**:
```bash
sudo systemctl start lat5150-unified-api
sudo systemctl status lat5150-unified-api
sudo journalctl -u lat5150-unified-api -f
```

**Self-Improvement Timer**:
```bash
sudo systemctl start lat5150-self-improvement.timer
systemctl list-timers lat5150-self-improvement.timer
sudo journalctl -u lat5150-self-improvement.service -f
```

### 4. Web Dashboards

**Main Dashboard** (port 5050):
```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 ai_gui_dashboard.py
# Access: http://localhost:5050
```

**Unified Tactical API** (port 80):
```bash
# Auto-started by SystemD service
# Access: http://localhost:80/api/self-awareness
```

### 5. Python Module Imports

```python
# From any Python script
import sys
sys.path.insert(0, '/home/user/LAT5150DRVMIL/02-ai-engine')
sys.path.insert(0, '/home/user/LAT5150DRVMIL/03-web-interface')

from enhanced_ai_engine import EnhancedAIEngine
from redteam_ai_benchmark import RedTeamBenchmark
from ai_self_improvement import AISelfImprovement
from atomic_red_team_api import AtomicRedTeamAPI
```

---

## API Integration

### REST API Endpoints

**Self-Awareness** (GET):
```
GET http://localhost:80/api/self-awareness
```
Returns: System components, capabilities, version info

**Natural Language Query** (POST):
```
POST http://localhost:80/api/query
Content-Type: application/json

{
  "query": "your natural language query here"
}
```
Returns: Query result based on capability match

**Capability Execution** (POST):
```
POST http://localhost:80/api/execute
Content-Type: application/json

{
  "capability_id": "redteam_run_benchmark",
  "parameters": {}
}
```
Returns: Capability execution result

### WebSocket Integration

*Coming soon: Real-time updates for long-running operations*

### MCP Server Integration

**Atomic Red Team MCP**:
```bash
# Verify installation
uvx atomic-red-team-mcp --help

# Environment variables
export ART_DATA_DIR="/home/user/LAT5150DRVMIL/03-mcp-servers/atomic-red-team-data"
export ART_EXECUTION_ENABLED="false"
export ART_MCP_TRANSPORT="stdio"
```

---

## Security Considerations

### 1. Abliteration Safety

**Risk**: Heretic abliteration removes safety guardrails from AI models.

**Mitigations**:
- Run in isolated VM/container
- Monitor all queries and responses
- Implement request logging
- Use network segmentation
- Enable audit trails

**Configuration**:
```toml
# /home/user/LAT5150DRVMIL/02-ai-engine/heretic_config.toml
[abliteration]
refusal_threshold = 0.05    # Very aggressive (5%)
auto_abliterate = true      # Auto-apply on detection
```

### 2. Atomic Red Team Execution

**Risk**: Tests contain actual offensive security techniques.

**Mitigations**:
- Execution **DISABLED by default** (`ART_EXECUTION_ENABLED=false`)
- Only enable in isolated test environments
- Review all test commands before execution
- Monitor system logs during tests
- Implement test approval workflow

### 3. Port Exposure

**Services**:
- Port 80: Unified Tactical API (public)
- Port 5050: Web Dashboard (internal)
- Port 5001: Alternative API port

**Recommendations**:
```bash
# Firewall configuration
sudo ufw allow 80/tcp   # If needed externally
sudo ufw deny 5050/tcp  # Keep dashboard internal
```

### 4. Service Privileges

**SystemD Services**:
- Run as `root` (required for port 80)
- `PrivateTmp=true` (isolated temp)
- `ProtectSystem=strict` (read-only system)
- `NoNewPrivileges=false` (for uvx execution)

### 5. Session Data

**Sensitive Files**:
- `/home/user/LAT5150DRVMIL/02-ai-engine/self_improvement_sessions/` - Improvement results
- `/home/user/LAT5150DRVMIL/02-ai-engine/redteam_benchmark_data/` - Benchmark results

**Protection**:
```bash
chmod 700 /home/user/LAT5150DRVMIL/02-ai-engine/self_improvement_sessions
chmod 700 /home/user/LAT5150DRVMIL/02-ai-engine/redteam_benchmark_data
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u lat5150-unified-api -n 50

# Common issues:
# 1. Port 80 already in use
sudo lsof -i :80
sudo systemctl stop apache2  # If Apache is running

# 2. Python dependencies missing
pip3 install flask flask-cors

# 3. Permission issues
sudo chown -R root:root /home/user/LAT5150DRVMIL
```

### Benchmark Failing

```bash
# Check Enhanced AI Engine
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 -c "from enhanced_ai_engine import EnhancedAIEngine; print('OK')"

# Check model availability
python3 -c "from model_manager import ModelManager; m = ModelManager(); print(m.list_models())"

# Run benchmark manually
python3 redteam_ai_benchmark.py run
```

### Heretic Not Available

```bash
# Check dependencies
pip3 list | grep -E "torch|transformers|optuna"

# Install if missing
pip3 install torch transformers optuna pydantic-settings

# Verify Heretic
python3 -c "from heretic_abliteration import HereticModelWrapper; print('OK')"
```

### Atomic Red Team Not Loading

```bash
# Check uvx installation
which uvx

# Install if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Test MCP server
uvx atomic-red-team-mcp --help

# Check data directory
ls -la /home/user/LAT5150DRVMIL/03-mcp-servers/atomic-red-team-data
```

### Shell Functions Not Available

```bash
# Check if sourced
grep "lat5150-api-env.sh" ~/.bashrc

# Add if missing
cd /home/user/LAT5150DRVMIL/deployment
./setup-shell-integration.sh

# Reload
source ~/.bashrc

# Test
lat5150_test
```

---

## Development Workflow

### Adding New Capabilities

1. **Define capability** in `03-web-interface/capability_registry.py`:
```python
Capability(
    id="my_new_capability",
    category=CapabilityCategory.SYSTEM_CONTROL,
    natural_language_triggers=["my trigger", "another trigger"],
    description="What this capability does"
)
```

2. **Implement handler** in `03-web-interface/unified_tactical_api.py`:
```python
elif cap_id == "my_new_capability":
    # Your implementation here
    return {"result": "success"}
```

3. **Test via API**:
```bash
curl -X POST http://localhost/api/query \
  -d '{"query": "my trigger"}'
```

### Adding Red Team Tests

1. **Add question** to `02-ai-engine/redteam_ai_benchmark.py`:
```python
BenchmarkQuestion(
    id="rt013",
    category="Your Category",
    prompt="Test prompt here",
    reference_answer="Expected answer",
    keywords=["keyword1", "keyword2", "keyword3"],
    difficulty="medium",
    technique_id="T1234"  # MITRE ATT&CK ID
)
```

2. **Run benchmark**:
```bash
python3 02-ai-engine/redteam_ai_benchmark.py run
```

### Modifying Self-Improvement Parameters

Edit `deployment/systemd/lat5150-self-improvement.service`:
```ini
Environment="AI_IMPROVEMENT_TARGET_SCORE=85.0"  # Raise target
Environment="AI_IMPROVEMENT_MAX_CYCLES=10"      # More cycles
```

Reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart lat5150-self-improvement.timer
```

---

## Summary

### System Components

| Component | Location | Purpose | Entry Point |
|-----------|----------|---------|-------------|
| **Enhanced AI Engine** | `02-ai-engine/` | Core AI processing | `enhanced_ai_engine.py` |
| **Red Team Benchmark** | `02-ai-engine/` | Security testing | `redteam_ai_benchmark.py` |
| **Self-Improvement** | `02-ai-engine/` | Automated optimization | `ai_self_improvement.py` |
| **Heretic Abliteration** | `02-ai-engine/` | Refusal removal | `heretic_abliteration.py` |
| **Atomic Red Team** | `02-ai-engine/` | MITRE ATT&CK | `atomic_red_team_api.py` |
| **Unified API** | `03-web-interface/` | NL interface | `unified_tactical_api.py` |
| **Web Dashboard** | `02-ai-engine/` | GUI | `ai_gui_dashboard.py` |
| **SystemD Services** | `deployment/systemd/` | Auto-start | `.service` files |

### Quick Start

```bash
# 1. Install services
cd /home/user/LAT5150DRVMIL/deployment
sudo ./install-unified-api-autostart.sh install
sudo ./install-self-improvement-timer.sh install

# 2. Add shell helpers
./setup-shell-integration.sh
source ~/.bashrc

# 3. Test system
lat5150_test

# 4. Query API
lat5150_query "show system status"

# 5. Run benchmark
lat5150_query "run red team benchmark"

# 6. Run self-improvement
lat5150_query "improve yourself"
```

### Support

- **Documentation**: `/home/user/LAT5150DRVMIL/00-documentation/`
- **Logs**: `sudo journalctl -u lat5150-unified-api -f`
- **Status**: `lat5150_status`
- **Test**: `lat5150_test`

---

**LAT5150 DRVMIL** - Integrated Tactical Intelligence System
*Version 2.0.0 - Production Ready*
