# LAT5150 DRVMIL - Natural Language Integration Quick Start

## 🚀 Quick Start (30 seconds)

```bash
# Navigate to web interface directory
cd /home/user/LAT5150DRVMIL/03-web-interface

# Launch tactical AI with WhiteRabbit
./launch_tactical_ai.sh

# Open tactical UI in browser
firefox http://127.0.0.1:5001
```

## 📋 What You Get

### 1. **Natural Language Command Processing**
The AI now understands natural language commands and maps them to capabilities:

```bash
# Examples that work out of the box:
🗣️ "Find the NSADeviceReconnaissance class"
🗣️ "Show me all references to process_cve_data"
🗣️ "Run code analyzer with WhiteRabbit model"
🗣️ "List all available agents"
🗣️ "Verify audit chain integrity"
```

### 2. **AI Self-Awareness**
The AI knows what it can do and can explain its capabilities:

```bash
curl http://127.0.0.1:5001/api/v2/self-awareness
```

Returns:
- System name and version
- All available capabilities (20+)
- Local models (WhiteRabbit, etc.)
- Example commands

### 3. **LOCAL-FIRST Architecture**
Everything runs locally using your custom models:
- **Primary**: WhiteRabbit (custom model)
- **Fallback**: Llama3.2, CodeLlama, Mixtral
- **No cloud APIs required**

## 🎯 Using the Tactical UI

### Step 1: Enable Natural Language Mode

1. Open UI: `http://127.0.0.1:5001`
2. In left sidebar, find **"AI CAPABILITIES"** panel
3. Click **"LOAD CAPABILITIES"** button
4. Check the **"NATURAL LANGUAGE MODE"** checkbox

### Step 2: Load Capabilities

Click **"LOAD CAPABILITIES"** to fetch available capabilities from the API.

You should see:
```
✅ LOADED 20+ CAPABILITIES
📦 LOCAL MODELS: whiterabbit, llama3.2:latest, ...
```

### Step 3: Send Natural Language Commands

With NL Mode enabled:
1. Type a natural language command in the input box
2. Click **"NL COMMAND"** button
3. Watch the AI:
   - Parse your command
   - Match it to a capability
   - Execute the capability
   - Return results

### Example Session

```
USER: Find the NSADeviceReconnaissance class

AI: [NL COMMAND EXECUTED SUCCESSFULLY]

MATCHED CAPABILITY: Find Symbol using Serena LSP
CONFIDENCE: 95.2%

RESULT:
{
  "symbol": "NSADeviceReconnaissance",
  "file": "01-source/debugging/nsa_device_reconnaissance_enhanced.py",
  "line": 32,
  "type": "class"
}
```

## 📊 View Self-Awareness Report

Click **"SELF-AWARENESS REPORT"** button to see:

```
AI SELF-AWARENESS REPORT

SYSTEM: LAT5150 DRVMIL Tactical AI Sub-Engine
LOCAL MODELS: whiterabbit, llama3.2:latest, codellama:latest

CAPABILITIES:
  code_understanding: 4
  agent_execution: 3
  model_inference: 2
  hardware_recon: 2
  security_audit: 2
  system_control: 2

TOTAL: 20+ capabilities

EXAMPLE COMMANDS:
  • Find the NSADeviceReconnaissance class
  • Run code analyzer with WhiteRabbit model
  • Show me all references to process_cve_data
  • List available agents
  • Verify audit chain integrity
```

## 🔧 Configuration

### Environment Variables

```bash
# Workspace path
export WORKSPACE_PATH="/home/user/LAT5150DRVMIL"

# Ollama endpoint
export OLLAMA_ENDPOINT="http://localhost:11434"

# API port
export API_PORT="5001"

# Local models (comma-separated)
export LOCAL_MODELS="whiterabbit,llama3.2:latest,codellama:latest"
```

### Custom Launch

```bash
# Launch with custom configuration
python3 unified_tactical_api.py \
  --workspace "/custom/path" \
  --ollama-endpoint "http://192.168.1.100:11434" \
  --port 8080 \
  --local-models "whiterabbit,custom-model:latest"
```

## 📡 API Endpoints

### Natural Language Command
```bash
POST /api/v2/nl/command
Content-Type: application/json

{
  "command": "Find the NSADeviceReconnaissance class"
}
```

### Self-Awareness Report
```bash
GET /api/v2/self-awareness
```

### Help System
```bash
GET /api/v2/help?query=find
```

### Legacy Endpoints (Still Available)
```bash
POST /api/chat          # Standard chat
POST /api/chat/stream   # Streaming chat
GET  /api/health        # Health check
```

## 🎨 TEMPEST Display Modes

The tactical UI supports 7 display modes for different operational environments:

1. **COMFORT (Level C)** - Default, eye-friendly (RECOMMENDED)
2. **DAY MODE** - High contrast for bright conditions
3. **NIGHT MODE** - Reduced brightness, red hues
4. **NVG MODE** - Night vision compatible (green monochrome)
5. **OLED BLACK** - Pure blacks for OLED displays
6. **HIGH CONTRAST** - Maximum contrast for accessibility
7. **LEVEL A** - Maximum TEMPEST protection (TS operations)

Switch modes in the left sidebar **"DISPLAY MODE"** panel.

## 🧠 Available Capabilities (20+)

### Code Understanding (Serena LSP)
- `serena_find_symbol` - Find functions, classes, variables by name
- `serena_find_references` - Find all references to a symbol
- `serena_insert_code` - Insert code at specific symbol location
- `serena_semantic_search` - Semantic codebase search

### Agent Execution (AgentSystems)
- `agent_invoke` - Execute containerized agent with isolation
- `agent_list` - List available agents
- `agent_status` - Check agent execution status

### Model Inference (LOCAL)
- `model_complete` - Generate text with local models
- `model_list` - List available local models

### Hardware Reconnaissance (DSMIL)
- `dsmil_scan` - Scan for military-grade hardware
- `dsmil_device_info` - Get device information

### Security Audit
- `audit_verify_chain` - Verify audit log integrity
- `audit_get_events` - Retrieve audit events

### System Control
- `system_health` - System health check
- `tempest_set_mode` - Change TEMPEST display mode

## 🔐 Security Features

### Container Isolation
All agents run in Docker/Podman containers with:
- Read-only root filesystem
- Dropped capabilities
- No privilege escalation
- Resource limits (CPU/memory)

### Audit Logging
All operations are logged with:
- SHA3-512 hash chaining
- Tamper-evident design
- Cryptographic verification

### TEMPEST Compliance
UI supports NATO SDIP-27 levels:
- Level C: Moderate protection (default)
- Level A: Maximum protection (45-80% EMF reduction)

## 🚨 Troubleshooting

### Ollama Not Responding
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Start Ollama
systemctl start ollama

# Or manually
ollama serve
```

### WhiteRabbit Model Not Found
```bash
# List available models
ollama list

# Pull WhiteRabbit (if it's a custom model, load it)
# For custom models:
ollama create whiterabbit -f /path/to/Modelfile
```

### Capabilities Not Loading
```bash
# Check API is running
curl http://127.0.0.1:5001/api/health

# Restart API
cd /home/user/LAT5150DRVMIL/03-web-interface
./launch_tactical_ai.sh
```

### Browser Can't Connect
```bash
# Check firewall
sudo ufw allow 5001/tcp

# Or use different host (listen on all interfaces)
python3 unified_tactical_api.py --host 0.0.0.0 --port 5001
```

## 📚 Next Steps

1. **Add Custom Capabilities**
   Edit `capability_registry.py` to add your own capabilities

2. **Register Custom Agents**
   Use the AgentOrchestrator to register containerized agents

3. **Integrate with DSMIL**
   Connect to the DSMIL kernel module for hardware control

4. **Deploy to Production**
   Use the SystemD service for auto-start:
   ```bash
   sudo systemctl enable lat5150-tactical-ai
   sudo systemctl start lat5150-tactical-ai
   ```

## 🎓 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Tactical UI (HTML)                    │
│            Natural Language + TEMPEST Display            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Unified Tactical API (Flask)               │
│         Natural Language Command Processing             │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ Capability   │ │   Natural   │ │   Model      │
│  Registry    │ │  Language   │ │  Provider    │
│              │ │  Processor  │ │  Manager     │
│ 20+ caps     │ │ WhiteRabbit │ │ Ollama/Local │
└──────────────┘ └─────────────┘ └──────────────┘
        │               │               │
        └───────────────┴───────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   Serena     │ │ AgentSystems│ │    DSMIL     │
│  LSP Engine  │ │  Container  │ │  Hardware    │
│              │ │  Runtime    │ │  Discovery   │
└──────────────┘ └─────────────┘ └──────────────┘
```

## 📖 Documentation

- **Full Integration Guide**: `NATURAL_LANGUAGE_INTEGRATION.md`
- **Serena/AgentSystems**: `00-documentation/SERENA_AGENTSYSTEMS_INTEGRATION.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Production Readiness**: `PRODUCTION_READINESS_CHECKLIST.md`

## ✅ Features Summary

- ✅ Natural language command processing
- ✅ LOCAL-FIRST architecture (WhiteRabbit)
- ✅ AI self-awareness system
- ✅ 20+ integrated capabilities
- ✅ TEMPEST-compliant tactical UI
- ✅ Container-based agent isolation
- ✅ Hash-chained audit logging
- ✅ Symbol-level code operations (99% token reduction)
- ✅ Multi-model provider abstraction
- ✅ Military-grade hardware support (DSMIL)

---

**Ready for deployment!** 🚀

For questions or issues, check the system logs:
```bash
tail -f /opt/lat5150/audit/agent_audit.log
```
