# DSMIL AI Engine - Hardware-Attested AI Inference

**Version:** 2.3.0
**Platform:** Dell Latitude 5450 MIL-SPEC
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## Overview

The DSMIL AI Engine provides hardware-attested AI inference with TPM binding and military-grade attestation. It integrates Ollama for local model execution with DSMIL security devices for tamper-evident operation.

### Key Features

- ✅ **Hardware Attestation** - TPM-backed inference verification
- ✅ **Multi-Model Routing** - Automatic model selection based on query complexity
- ✅ **Code Generation** - Specialized models for coding tasks
- ✅ **Zero External Dependencies** - 100% local inference
- ✅ **Military Mode Integration** - DSMIL device binding
- ✅ **Unified Web Interface** - HTTP API for all operations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Application                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼─────────┐    ┌─────────▼──────────┐
│  Unified Server   │    │  CLI Interface     │
│  (Port 9876)      │    │  dsmil_ai_engine   │
└─────────┬─────────┘    └─────────┬──────────┘
          │                         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Unified Orchestrator   │
          │  (Smart Routing)         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   DSMIL AI Engine        │
          │  (Attestation Layer)     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Ollama Service         │
          │  (localhost:11434)       │
          └────────────┬────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼────┐      ┌─────▼──────┐    ┌─────▼─────┐
│deepseek│      │deepseek-   │    │qwen2.5-   │
│-r1     │      │coder       │    │coder      │
│1.5b    │      │6.7b        │    │7b         │
└────────┘      └────────────┘    └───────────┘
  Fast            Code Gen         Quality Code
```

---

## Quick Start

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service (in separate terminal)
ollama serve
```

### 2. Pull AI Models

```bash
# Fast model (1.5GB) - General queries
ollama pull deepseek-r1:1.5b

# Code model (4GB) - Code generation
ollama pull deepseek-coder:6.7b-instruct

# Quality code model (4.5GB) - Complex code
ollama pull qwen2.5-coder:7b

# Uncensored code model (20GB) - No restrictions ✨
# Requires manual GGUF import from HuggingFace:
# 1. Download from: huggingface.co/TheBloke/WizardLM-1.0-Uncensored-CodeLlama-34B-GGUF
# 2. Create Modelfile: echo "FROM ./wizardlm-34b.Q4_K_M.gguf" > Modelfile
# 3. Import: ollama create wizardlm-uncensored-codellama:34b -f Modelfile

# Optional: Large model (40GB) - Code review
ollama pull codellama:70b
```

### 3. Start AI Server

```bash
# Use the automated startup script
cd /path/to/LAT5150DRVMIL/02-ai-engine
./start_ai_server.sh
```

The script will:
- ✓ Validate all dependencies
- ✓ Check Ollama service
- ✓ Verify models are available
- ✓ Start the unified web server
- ✓ Display access URL

### 4. Use the TUI Manager (Recommended)

```bash
# Interactive TUI for AI management
cd /path/to/LAT5150DRVMIL/02-ai-engine
python3 ai_tui.py
```

Features:
- ✅ Run AI queries interactively
- ✅ Configure guardrails (no guardrails by default)
- ✅ Manage models
- ✅ View system status
- ✅ Test all models
- ✅ Export/import configuration

---

## Usage

### TUI Manager (Interactive)

The easiest way to use the AI engine is through the interactive TUI:

```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine
python3 ai_tui.py
```

**TUI Features:**

```
╔══════════════════════════════════════════════════════════════╗
║           DSMIL AI ENGINE - TUI MANAGER                      ║
║      Hardware-Attested AI Inference Control                  ║
╚══════════════════════════════════════════════════════════════╝

  Ollama: 🟢 Connected | Models: 3/4 | Mode 5: STANDARD

  [1] Run AI Query               ← Interactive queries
  [2] Configure Guardrails       ← No guardrails (default)
  [3] Model Management           ← Install/manage models
  [4] System Status              ← Full status display
  [5] Settings                   ← Config management
  [6] Test Models                ← Test all available
  [0] Exit
```

**Guardrails Options:**
1. **No Guardrails** (Default) - Direct technical answers, no restrictions
2. **Basic Safety** - Refuse illegal/harmful requests only
3. **Corporate** - Professional, workplace-appropriate
4. **Educational** - Academic rigor, cite sources
5. **Security Focus** - Cybersecurity expert mode
6. **Custom** - Write your own system prompt

### Web Interface

Once the server is running, access the web interface at:

```
http://localhost:9876
```

Features:
- Chat interface with AI models
- Model selection (auto/fast/code/quality_code)
- Hardware attestation display
- System status monitoring

### API Endpoints

#### Chat / Query

```bash
# Auto-routed query
curl "http://localhost:9876/ai/chat?msg=explain%20TPM%20attestation"

# Force specific model
curl "http://localhost:9876/ai/chat?msg=write%20python%20function&model=code"
```

#### AI Status

```bash
curl http://localhost:9876/ai/status
```

Returns:
```json
{
  "ollama": {
    "url": "http://localhost:11434",
    "connected": true
  },
  "models": {
    "fast": {"name": "deepseek-r1:1.5b", "available": true},
    "code": {"name": "deepseek-coder:6.7b-instruct", "available": true},
    "quality_code": {"name": "qwen2.5-coder:7b", "available": true},
    "large": {"name": "codellama:70b", "available": false}
  },
  "dsmil": {
    "mode5": {
      "mode5_enabled": true,
      "mode5_level": "STANDARD",
      "safe": true,
      "devices_available": 84
    }
  }
}
```

### Command Line Interface

```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine

# Check status
python3 dsmil_ai_engine.py status

# Run query with auto model selection
python3 dsmil_ai_engine.py prompt "explain quantum computing"

# Force specific model
python3 dsmil_ai_engine.py prompt "write sorting algorithm" code

# Custom system prompt
python3 dsmil_ai_engine.py set-prompt "You are a security expert..."
python3 dsmil_ai_engine.py get-prompt
```

---

## Model Selection Strategy

The AI engine automatically routes queries to the optimal model:

| Model Tier | Model | Size | Use Case | Speed |
|------------|-------|------|----------|-------|
| **fast** | deepseek-r1:1.5b | 1.5GB | Quick questions, facts | ~5s |
| **code** | deepseek-coder:6.7b | 4GB | Code generation | ~10s |
| **quality_code** | qwen2.5-coder:7b | 4.5GB | Complex algorithms | ~15s |
| **uncensored_code** | wizardlm-uncensored-codellama:34b | 20GB | Uncensored code, no restrictions ✨ | ~30s |
| **large** | codellama:70b | 40GB | Code review, analysis | ~60s |

### Auto-Routing Logic

```
Query Length > 200 chars OR
Contains: "analyze", "explain", "research", "investigate"
    └─> Route to: large

Contains: "code", "function", "implement", "algorithm"
    └─> Route to: code or quality_code

Default:
    └─> Route to: fast
```

---

## Quantization Support

The AI engine supports model quantization to reduce memory usage and improve performance on memory-constrained systems.

### Quantization Levels

| Level | Precision | Size Reduction | Quality | Use Case |
|-------|-----------|----------------|---------|----------|
| **Q4_0** | 4-bit | ~75% smaller | Good | Memory-constrained systems |
| **Q4_K_M** | 4-bit mixed | ~70% smaller | Better | Recommended for most users |
| **Q5_K_M** | 5-bit mixed | ~65% smaller | Excellent | Balance of size and quality |
| **Q8_0** | 8-bit | ~50% smaller | Near-perfect | High-quality with space savings |

### Quantized Model Options

```bash
# Fast model (1.5GB -> 1GB)
ollama pull deepseek-r1:1.5b-q4_0

# Code model (4GB -> 2.5GB)
ollama pull deepseek-coder:6.7b-q4_K_M

# Quality code model (4.5GB -> 3GB)
ollama pull qwen2.5-coder:7b-q4_K_M

# Uncensored model (20GB -> 12GB)
# Create quantized version from GGUF Q4_K_M variant

# Large model options:
ollama pull codellama:70b-q4_K_M  # 40GB -> 25GB
ollama pull codellama:70b-q5_K_M  # 40GB -> 30GB (better quality)
```

### Automatic Quantization Recommendations

The AI engine automatically detects your system RAM and recommends appropriate models:

```bash
python3 dsmil_ai_engine.py recommend-quant
```

**Memory Guidelines:**
- < 8GB RAM: Use fast model only, consider Q4 quantization
- 8-16GB RAM: Use fast + code_q4 models
- 16-32GB RAM: Use fast + code + quality_code_q4 models
- 32-64GB RAM: Use all standard models + uncensored_q4
- > 64GB RAM: Can handle all models including large (consider Q4/Q5 for faster inference)

### Trade-offs

**Advantages of Quantization:**
- ✅ 50-75% less disk space
- ✅ 50-75% less RAM/VRAM usage
- ✅ Faster inference (less memory bandwidth)
- ✅ Can run larger models on smaller hardware

**Disadvantages:**
- ⚠️ Slight quality degradation (usually imperceptible with Q4_K_M and above)
- ⚠️ Not all models have official quantized versions

---

## Hardware Attestation

Every AI inference is attested using DSMIL security devices:

```json
{
  "response": "...",
  "attestation": {
    "dsmil_device": 12,
    "mode5_level": "STANDARD",
    "response_hash": "a3f5b8c1...",
    "verified": true
  }
}
```

### Attestation Chain

1. **Query Hashing** - SHA256 hash of user prompt
2. **TPM Sealing** - Bind inference to platform state (optional)
3. **Response Signing** - Cryptographic signature of output
4. **Verification** - Validate integrity before returning

---

## Configuration

### Dynamic Paths

All paths are automatically configured based on installation location:

```python
BASE_DIR = Path(__file__).parent.parent.resolve()  # LAT5150DRVMIL
AI_ENGINE_DIR = BASE_DIR / "02-ai-engine"
WEB_INTERFACE_DIR = BASE_DIR / "03-web-interface"
INTEGRATIONS_DIR = BASE_DIR / "04-integrations"
```

Works for:
- ✅ Any user (no hardcoded /home/john)
- ✅ Any installation path
- ✅ Multiple users/group environments
- ✅ Docker containers

### Custom System Prompt

Create custom system prompts for specialized tasks:

```bash
# Set custom prompt
python3 dsmil_ai_engine.py set-prompt "You are a malware analyst with expertise in APT detection..."

# Prompt saved to: ~/.claude/custom_system_prompt.txt
```

### Model Configuration

Edit `dsmil_ai_engine.py` to change models:

```python
self.models = {
    "fast": "deepseek-r1:1.5b",
    "code": "deepseek-coder:6.7b-instruct",
    "quality_code": "qwen2.5-coder:7b",
    "large": "codellama:70b"
}
```

---

## Troubleshooting

### Ollama Not Running

**Error:** `No AI models available`

**Solution:**
```bash
# Start Ollama
ollama serve

# Or use systemd
sudo systemctl start ollama
```

### Models Not Available

**Error:** `Models: 0/4 available`

**Solution:**
```bash
# Pull at least the fast model
ollama pull deepseek-r1:1.5b

# Check models
ollama list
```

### Connection Refused

**Error:** `Connection refused to localhost:11434`

**Check:**
```bash
# Is Ollama running?
ps aux | grep ollama

# Is port 11434 listening?
netstat -tulpn | grep 11434

# Try starting Ollama
ollama serve
```

### Slow Inference

**Issue:** Queries taking too long

**Solutions:**
- Use smaller models (deepseek-r1:1.5b)
- Enable GPU acceleration (CUDA/ROCm)
- Increase system RAM
- Check CPU load: `top`

### Path Errors

**Error:** `FileNotFoundError: /home/john/...`

**Solution:** Pull latest code - all hardcoded paths have been removed in v2.0.0

---

## Performance

### Inference Benchmarks

Tested on Dell Latitude 5450 (Intel Meteor Lake, 32GB RAM):

| Model | Query Type | Time | Tokens/sec |
|-------|------------|------|------------|
| deepseek-r1:1.5b | "What is TPM?" | 4.2s | 28.3 |
| deepseek-coder:6.7b | "Write Python function" | 9.8s | 15.7 |
| qwen2.5-coder:7b | "Implement sorting algo" | 14.1s | 12.4 |
| codellama:70b | "Review this code" | 58.3s | 4.1 |

### GPU Acceleration

To enable GPU acceleration:

```bash
# NVIDIA GPU
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# AMD ROCm
HSA_OVERRIDE_GFX_VERSION=10.3.0 ollama serve
```

---

## Security

### Threat Model

Protected against:
- ✅ Model tampering (TPM attestation)
- ✅ Prompt injection (input validation)
- ✅ Output manipulation (response signing)
- ✅ Man-in-the-middle (localhost only)

Not protected against:
- ❌ Physical hardware access
- ❌ Root-level compromise
- ❌ Model backdoors (use trusted sources)

### Hardening

```bash
# 1. Restrict to localhost only (default)
# Server listens on 127.0.0.1:9876

# 2. Enable TPM sealing
python3 -c "from dsmil_military_mode import DSMILMilitaryMode; d = DSMILMilitaryMode(); d.seal_model_weights('~/.ollama/models')"

# 3. Monitor audit log
tail -f /var/log/dsmil_ai_attestation.log
```

---

## Integration

### Python Integration

```python
from dsmil_ai_engine import DSMILAIEngine

engine = DSMILAIEngine()

# Run query
result = engine.generate("Explain TPM", model_selection="auto")

if 'response' in result:
    print(result['response'])
    print(f"Verified: {result['attestation']['verified']}")
```

### REST API Integration

```python
import requests

response = requests.get(
    'http://localhost:9876/ai/chat',
    params={'msg': 'Explain cryptographic attestation'}
)

data = response.json()
print(data['response'])
```

### Bash Integration

```bash
#!/bin/bash
# Query AI from bash script

RESPONSE=$(curl -s "http://localhost:9876/ai/chat?msg=list%20security%20tools")
echo $RESPONSE | jq -r '.response'
```

---

## Pydantic AI Integration (Type-Safe Mode)

**New in Version 2.3.0** - The AI engine now supports dual-mode operation with optional Pydantic AI integration for type-safe, validated AI inference.

### Why Pydantic AI?

Traditional AI responses are dictionaries with string keys - prone to typos and runtime errors:

```python
# ❌ Legacy dict mode - no type safety
result = engine.generate("query")
print(result['respons'])  # Typo! Runtime error
```

With Pydantic AI, responses are validated models with full IDE autocomplete:

```python
# ✅ Pydantic mode - type-safe
result = engine.generate(request)
print(result.response)  # Autocomplete works! Validated at runtime
```

### Features

- ✅ **Type Safety** - Full type checking with IDE autocomplete
- ✅ **Runtime Validation** - Automatic input/output validation
- ✅ **Structured Outputs** - Force LLM to return valid JSON schemas
- ✅ **Backward Compatible** - Legacy dict mode still works
- ✅ **Performance** - Minimal overhead (binary IPC 10-50x faster than Pydantic, use both strategically)

### Installation

```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine

# Install Pydantic AI (optional - engine works without it)
./install_pydantic_ai.sh

# Or manually:
pip install pydantic pydantic-ai ollama
```

The installation script:
- ✓ Checks Python version (≥3.9 required)
- ✓ Installs dependencies from requirements-pydantic-ai.txt
- ✓ Verifies Ollama service status
- ✓ Tests imports and engine creation

### Usage

#### Mode 1: Legacy Dict Mode (Backward Compatible)

```python
from dsmil_ai_engine import DSMILAIEngine

# Create engine in legacy mode (default)
engine = DSMILAIEngine(pydantic_mode=False)

# Use string prompts, get dict responses
result = engine.generate("What is TPM attestation?", model_selection="fast")

# Access with dict keys
if result.get('success'):
    print(result['response'])
    print(f"Model: {result['model']}")
```

#### Mode 2: Pydantic Type-Safe Mode

```python
from dsmil_ai_engine import DSMILAIEngine
from pydantic_models import DSMILQueryRequest, ModelTier

# Create engine in Pydantic mode
engine = DSMILAIEngine(pydantic_mode=True)

# Create type-safe request
request = DSMILQueryRequest(
    prompt="Explain kernel module compilation",
    model=ModelTier.FAST,
    temperature=0.7
)

# Get validated Pydantic response
result = engine.generate(request)

# Access with type-safe properties (IDE autocomplete!)
print(result.response)  # No typos possible
print(f"Model: {result.model_used}")
print(f"Latency: {result.latency_ms:.2f}ms")
print(f"Confidence: {result.confidence:.2f}")
```

#### Mode 3: Hybrid Mode (Per-Call Override)

```python
from dsmil_ai_engine import DSMILAIEngine

# Create engine in legacy mode
engine = DSMILAIEngine(pydantic_mode=False)

# Call 1: Use default (dict)
result1 = engine.generate("Hello", model_selection="fast")
# Returns dict

# Call 2: Override to Pydantic for this call
result2 = engine.generate("Hello", model_selection="fast", return_pydantic=True)
# Returns DSMILQueryResult (Pydantic model)
```

### CLI Usage with Pydantic

```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine

# Check Pydantic availability
python3 ai.py
# Shows: "Pydantic AI: Available" or "Not installed"

# Use type-safe mode
python3 ai.py --pydantic "What is TPM attestation?"

# Interactive type-safe mode
python3 ai.py -i --pydantic

# Combine with other flags
python3 ai.py -c --pydantic --verbose "Write Python function"
```

### Available Pydantic Models

The integration includes comprehensive type-safe models:

```python
from pydantic_models import (
    # Core AI models
    DSMILQueryRequest,    # Type-safe query input
    DSMILQueryResult,     # Validated AI response

    # Specialized outputs
    CodeGenerationResult,     # Structured code with metadata
    SecurityAnalysisResult,   # Security findings
    MalwareAnalysisResult,    # Malware classification

    # Configuration
    ModelTier,           # Enum: FAST, CODE, QUALITY_CODE, etc.
    AIEngineConfig,      # Engine configuration with validation

    # Agent orchestration
    AgentTaskRequest,    # Agent task input
    AgentTaskResult,     # Agent execution result
)
```

### Performance Comparison

Benchmark results (10,000 iterations):

| Method | Serialize | Deserialize | Total | Use Case |
|--------|-----------|-------------|-------|----------|
| **Binary (struct)** | 1.2μs | 0.8μs | 2.0μs | Ultra-low latency IPC, agent communication |
| **JSON Dict** | 8.5μs | 6.2μs | 14.7μs | Legacy compatibility |
| **Pydantic** | 42.1μs | 28.3μs | 70.4μs | Type safety, validation, web APIs |

**Recommendation:**
- Use **Binary** for: Agent IPC, real-time streams, performance-critical paths
- Use **Pydantic** for: Web APIs, type safety, developer experience, validation

Run benchmark yourself:
```bash
python3 benchmark_binary_vs_pydantic.py
```

### Examples

See comprehensive examples in:
- `example_pydantic_usage.py` - 4 usage patterns demonstrated
- `test_dual_mode.py` - Full test suite (requires Ollama)
- `test_imports.py` - Quick import test (no Ollama needed)

```bash
# Run examples
python3 example_pydantic_usage.py

# Test both modes
python3 test_dual_mode.py

# Quick import test
python3 test_imports.py
```

### Structured Code Generation

Force the LLM to return validated, structured code:

```python
from dsmil_ai_engine import DSMILAIEngine
from pydantic_models import CodeGenerationResult, ModelTier

engine = DSMILAIEngine(pydantic_mode=True)

# Request structured code output
request = DSMILQueryRequest(
    prompt="Create a secure password hashing function",
    model=ModelTier.CODE
)

result = engine.generate(request, use_structured_output=True)

# Result is CodeGenerationResult with validation
print(result.code)              # Generated code
print(result.language)          # Validated: python|rust|c|cpp|bash
print(result.explanation)       # What the code does
print(result.security_notes)    # Security considerations
print(result.dependencies)      # Required packages
```

### Benefits Summary

| Aspect | Legacy Dict | Pydantic AI |
|--------|-------------|-------------|
| **Type Safety** | ❌ No | ✅ Full IDE support |
| **Validation** | ❌ Manual | ✅ Automatic |
| **Autocomplete** | ❌ No | ✅ Yes |
| **Error Detection** | 🟡 Runtime only | ✅ IDE + Runtime |
| **Structured Output** | ❌ String parsing | ✅ Validated models |
| **Testing** | 🟡 Complex mocking | ✅ Easy model testing |
| **API Docs** | ❌ Manual | ✅ Auto-generated |
| **Performance** | ✅ Fast | 🟡 Slightly slower (70μs vs 15μs) |

### Documentation

For detailed integration information, see:
- `PYDANTIC_AI_INTEGRATION.md` - Full integration plan and migration guide
- `pydantic_models.py` - All available type-safe models
- `dsmil_ai_engine_v2.py` - Next-gen Pydantic AI engine (async)

---

## Files

```
02-ai-engine/
├── ai_tui.py                       # Interactive TUI manager ✨
├── ai.py                           # Clean CLI interface
├── dsmil_ai_engine.py              # Main AI engine (dual-mode support) ✨
├── dsmil_ai_engine_v2.py           # Next-gen Pydantic AI engine (async)
├── dsmil_military_mode.py          # Hardware attestation
├── unified_orchestrator.py         # Multi-backend routing
├── start_ai_server.sh              # Startup validator script
│
├── pydantic_models.py              # Type-safe Pydantic models ✨ NEW
├── example_pydantic_usage.py       # Pydantic usage examples ✨ NEW
├── benchmark_binary_vs_pydantic.py # Performance comparison ✨ NEW
├── install_pydantic_ai.sh          # Pydantic AI installer ✨ NEW
├── requirements-pydantic-ai.txt    # Pydantic dependencies ✨ NEW
├── test_dual_mode.py               # Dual-mode test suite ✨ NEW
├── test_imports.py                 # Import validation tests ✨ NEW
│
├── PYDANTIC_AI_INTEGRATION.md      # Integration guide ✨ NEW
├── README.md                       # This file
└── sub_agents/
    └── openai_wrapper.py           # OpenAI API compatibility

03-web-interface/
└── dsmil_unified_server.py         # Web server (port 9876)

04-integrations/
├── rag_manager.py                  # RAG system integration
├── crawl4ai_wrapper.py             # Web scraping
└── web_archiver.py                 # Research archive
```

---

## Changelog

### Version 2.3.0 (2025-11-19) - Pydantic AI Integration
- ✅ Added Pydantic AI support for type-safe inference
- ✅ Dual-mode operation: dict (legacy) and Pydantic (type-safe)
- ✅ Type-safe request/response models with validation
- ✅ Enhanced CLI with --pydantic flag
- ✅ Structured code generation with CodeGenerationResult
- ✅ Security analysis with SecurityAnalysisResult
- ✅ Performance benchmark: Binary vs Pydantic
- ✅ Comprehensive test suite and examples
- ✅ Installation script with verification
- ✅ Backward compatible - legacy mode still works

### Version 2.2.0 (2025-11-06)
- ✅ Added WizardLM-1.0-Uncensored-CodeLlama-34b model support
- ✅ MSR (Model-Specific Register) hardware access
- ✅ SMM (System Management Mode) capability
- ✅ Memory-Mapped I/O (MMIO) support
- ✅ Device operation discovery tool
- ✅ Firmware-level operation support

### Version 2.1.0 (2025-11-06)
- ✅ Added interactive TUI manager (ai_tui.py)
- ✅ Configurable guardrails system (6 presets + custom)
- ✅ Default: No guardrails mode
- ✅ Interactive query interface
- ✅ Model management interface
- ✅ Configuration export/import

### Version 2.0.0 (2025-11-06)
- ✅ Removed all hardcoded paths (37 instances)
- ✅ Added dynamic path configuration
- ✅ Created startup validator script
- ✅ Improved error handling
- ✅ Added comprehensive documentation
- ✅ Multi-user/group support

### Version 1.5.0 (2025-01-05)
- Multi-model routing
- Code generation support
- Custom system prompts
- Hardware attestation

---

## 📚 Documentation Organization

**IMPORTANT**: All .md documentation files MUST be organized into the `docs/` directory structure.

### Documentation Structure

```
02-ai-engine/
├── README.md (this file - main entry point)
└── docs/
    ├── INDEX.md (documentation index)
    ├── context-optimization/ (context window optimization docs)
    ├── integrations/ (integration guides and completion reports)
    ├── setup-guides/ (setup and configuration guides)
    ├── features/ (feature documentation)
    ├── research/ (research insights)
    └── benchmarking/ (performance and optimization)
```

### Documentation Standards

✅ **ALWAYS** organize new .md files into appropriate `docs/` subdirectories
✅ **ALWAYS** update `docs/INDEX.md` when adding new documentation
✅ **ALWAYS** use descriptive, consistent file naming
✅ **NEVER** leave .md files in the root directory (except README.md)
✅ **ALWAYS** cross-reference related documents

### Quick Links

- **Documentation Index**: [`docs/INDEX.md`](docs/INDEX.md)
- **Context Optimization**: [`docs/context-optimization/CONTEXT_OPTIMIZATION_README.md`](docs/context-optimization/CONTEXT_OPTIMIZATION_README.md)
- **Setup Guides**: [`docs/setup-guides/MCP_SERVER_GUIDE.md`](docs/setup-guides/MCP_SERVER_GUIDE.md)
- **Integration**: [`docs/integrations/README_INTEGRATION.md`](docs/integrations/README_INTEGRATION.md)

**See [`docs/INDEX.md`](docs/INDEX.md) for complete documentation catalog.**

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Run diagnostics: `./start_ai_server.sh`
3. Check logs: `/var/log/dsmil_ai_attestation.log`
4. Review Ollama logs: `journalctl -u ollama`
5. Review documentation: `docs/INDEX.md`

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Last Updated:** 2025-11-19
**Version:** 2.3.0
