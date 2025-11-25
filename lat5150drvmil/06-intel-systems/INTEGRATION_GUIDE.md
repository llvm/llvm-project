# Screenshot Intelligence Integration Guide

Complete guide for integrating Screenshot Intelligence with LAT5150DRVMIL AI subsystems

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [DSMIL AI Engine Integration](#dsmil-ai-engine-integration)
3. [Unified Orchestrator Integration](#unified-orchestrator-integration)
4. [MCP Server Integration](#mcp-server-integration)
5. [RAG System Integration](#rag-system-integration)
6. [OSINT Integration (Telegram/Signal)](#osint-integration)
7. [SWORD Intelligence Integration](#sword-intelligence-integration)
8. [Complete Workflow Examples](#complete-workflow-examples)

---

## Integration Overview

The Screenshot Intelligence System integrates with LAT5150DRVMIL AI subsystems at multiple levels:

```
┌─────────────────────────────────────────────────────────────┐
│              LAT5150DRVMIL AI Platform                       │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Unified          │         │ DSMIL AI Engine  │         │
│  │ Orchestrator     │◄────────┤ (5 Models)       │         │
│  └────────┬─────────┘         └──────────────────┘         │
│           │                                                  │
│           ├──────────┬──────────┬──────────────┐           │
│           │          │          │              │           │
│  ┌────────▼──────┐ ┌▼──────┐ ┌▼────────┐ ┌───▼────────┐  │
│  │ Screenshot    │ │ RAG   │ │ OSINT   │ │ Security   │  │
│  │ Intelligence  │ │ System│ │ Intel   │ │ Tools      │  │
│  └───────────────┘ └───────┘ └─────────┘ └────────────┘  │
│                                                              │
│  MCP Servers (12): dsmil-ai, screenshot-intelligence, ...  │
└─────────────────────────────────────────────────────────────┘
```

**Key Integration Points:**
- **MCP Server**: 12th MCP server added to unified orchestrator
- **RAG System**: Vector database upgrade (51.8% → 88%+ accuracy)
- **DSMIL AI**: Used for summarization and analysis
- **OSINT**: Telegram/Signal message correlation
- **SWORD**: Forensic timeline and intelligence operations

---

## DSMIL AI Engine Integration

### Overview

Screenshot Intelligence uses DSMIL AI Engine for:
- Event summarization
- Incident analysis
- Content classification
- Anomaly description

### Integration Code

```python
from dsmil_ai_engine import DSMILAIEngine
from ai_analysis_layer import AIAnalysisLayer

# Initialize with DSMIL AI
ai_analysis = AIAnalysisLayer(use_ai_engine=True)

# The AI Analysis Layer automatically uses DSMIL for:
# - generate_incident_summary()
# - Advanced content analysis
# - Pattern description
```

### Example: AI-Powered Incident Summary

```python
from screenshot_intelligence import ScreenshotIntelligence
from ai_analysis_layer import AIAnalysisLayer
from datetime import datetime

# Initialize
intel = ScreenshotIntelligence()
ai_analysis = AIAnalysisLayer(screenshot_intel=intel, use_ai_engine=True)

# Analyze timeline
results = ai_analysis.analyze_timeline(
    start_time=datetime(2025, 11, 10),
    end_time=datetime(2025, 11, 12),
    auto_detect_incidents=True
)

# Generate AI summary for each incident
for incident in results['incidents']:
    summary = ai_analysis.generate_incident_summary(incident, max_length=500)
    print(f"\nIncident: {incident.incident_name}")
    print(f"Summary: {summary}")
```

### Model Selection

```python
# Use different DSMIL models for different tasks

# Fast model for quick summaries
ai_engine.query(prompt, model="fast")  # deepseek-r1:1.5b

# Code model for technical analysis
ai_engine.query(prompt, model="code")  # deepseek-coder:6.7b

# Quality model for comprehensive analysis
ai_engine.query(prompt, model="quality_code")  # qwen2.5-coder:7b

# Uncensored model (default) for unrestricted analysis
ai_engine.query(prompt, model="uncensored_code")  # wizardlm-uncensored-codellama:34b
```

---

## Unified Orchestrator Integration

### Overview

Screenshot Intelligence integrates with `unified_orchestrator.py` through the MCP server.

### Configuration

Already added to `02-ai-engine/mcp_servers_config.json`:

```json
{
  "screenshot-intelligence": {
    "command": "python3",
    "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/screenshot_intel_mcp_server.py"],
    "env": {
      "PYTHONPATH": "/home/user/LAT5150DRVMIL:/home/user/LAT5150DRVMIL/04-integrations/rag_system"
    },
    "description": "Screenshot Intelligence with OCR, Vector RAG, timeline analysis..."
  }
}
```

### Using via Orchestrator

```python
from unified_orchestrator import UnifiedAIOrchestrator

# Initialize orchestrator (includes all 12 MCP servers)
orchestrator = UnifiedAIOrchestrator()

# Query that triggers screenshot intelligence
response = orchestrator.query(
    "Find screenshots containing VPN errors from last week",
    force_backend="local"  # Uses DSMIL AI + Screenshot Intel MCP
)
```

### MCP Tools Available

1. **ingest_screenshot** - Ingest single screenshot
2. **scan_device** - Batch scan device directory
3. **search_intel** - Semantic search
4. **timeline_query** - Query by date range
5. **generate_timeline_report** - Generate reports
6. **register_device** - Device management
7. **get_stats** - System statistics

---

## MCP Server Integration

### Starting MCP Server

```bash
# Via unified orchestrator (automatic)
# The screenshot-intelligence MCP server starts when orchestrator is initialized

# Manual start for testing
python3 /home/user/LAT5150DRVMIL/02-ai-engine/screenshot_intel_mcp_server.py
```

### Using MCP Tools

```python
# Example MCP tool call structure
{
  "tool": "search_intel",
  "arguments": {
    "query": "database error",
    "limit": 10,
    "doc_type": "image"
  }
}

# Response format
{
  "query": "database error",
  "total_results": 5,
  "results": [
    {
      "score": 0.87,
      "type": "image",
      "timestamp": "2025-11-12T14:30:22",
      "text_preview": "Error connecting to database...",
      "metadata": {...}
    }
  ]
}
```

---

## RAG System Integration

### Overview

Screenshot Intelligence **extends** the existing RAG system with:
- Vector database (Qdrant)
- Transformer embeddings (BAAI/bge-base-en-v1.5)
- Screenshot/image support
- Chat message support

### Backward Compatibility

The old TF-IDF RAG system (`04-integrations/rag_system.py`) still works. New vector RAG is in `vector_rag_system.py`.

### Migration Path

```python
# Old RAG system (TF-IDF)
from rag_system import RAGSystem
old_rag = RAGSystem()

# New Vector RAG system
from vector_rag_system import VectorRAGSystem
new_rag = VectorRAGSystem()

# API is compatible! Just better accuracy
results = new_rag.search("query", limit=10)  # 88%+ accuracy vs 51.8%
```

### Adding Screenshots to RAG

```python
from vector_rag_system import VectorRAGSystem

rag = VectorRAGSystem()

# Ingest screenshot (automatic OCR)
result = rag.ingest_document(
    filepath="/path/to/screenshot.png",
    doc_type='image',
    metadata={'device': 'phone1'}
)

# Search across ALL content (docs + screenshots)
results = rag.search("VPN error", limit=10)
```

---

## OSINT Integration

### Telegram Integration

```python
from telegram_integration import TelegramIntegration, TelegramConfig
from vector_rag_system import VectorRAGSystem

# Configure Telegram
config = TelegramConfig(
    api_id="YOUR_API_ID",
    api_hash="YOUR_API_HASH",
    phone_number="+1234567890"
)

# Initialize (shares same Vector RAG)
rag = VectorRAGSystem()
telegram = TelegramIntegration(config, vector_rag=rag)

# Start client
await telegram.start()

# Add monitored chat
await telegram.add_monitored_chat('@channel_username')

# Sync messages to vector RAG
stats = await telegram.sync_chat_history('chat_id', limit=1000)

# Now screenshots AND telegram messages are in same vector DB!
# Search across both
results = rag.search("project update", limit=10)
```

### Signal Integration

```python
from signal_integration import SignalIntegration, SignalConfig

# Configure Signal
config = SignalConfig(
    phone_number="+1234567890"
)

# Initialize (shares same Vector RAG)
signal = SignalIntegration(config, vector_rag=rag)

# Sync messages
stats = signal.sync_messages(max_messages=1000)

# Search across screenshots + Telegram + Signal
results = rag.search("meeting notes", limit=10)
```

### Timeline Correlation

```python
from datetime import datetime

# Get all events from all sources for a day
events = rag.timeline_query(
    start_time=datetime(2025, 11, 12, 0, 0),
    end_time=datetime(2025, 11, 12, 23, 59)
)

# Events include:
# - Screenshots (from phones/laptops)
# - Telegram messages
# - Signal messages
# - PDFs, docs, etc.

# All chronologically ordered with correlation
```

---

## SWORD Intelligence Integration

### Overview

Screenshot Intelligence is designed for SWORD Intelligence operations:
- Forensic timeline analysis
- Evidence preservation with TPM attestation
- Multi-source intelligence correlation
- Local-first (no cloud exposure)

### Forensic Timeline Export

```python
from screenshot_intelligence import ScreenshotIntelligence
from datetime import datetime
import json

intel = ScreenshotIntelligence()

# Generate forensic timeline report
report = intel.generate_timeline_report(
    start_time=datetime(2025, 11, 10),
    end_time=datetime(2025, 11, 12),
    output_format='json'
)

# Export for legal proceedings
with open('evidence_timeline.json', 'w') as f:
    json.dump(report, f, indent=2)

# Timeline includes:
# - Exact timestamps
# - Source attribution (device/app)
# - Content hashes (integrity verification)
# - OCR confidence scores
# - Correlation links
```

### TPM Attestation

```python
# When DSMIL is loaded, vector RAG responses can be TPM-attested
from dsmil_military_mode import DSMILMilitaryMode

dsmil = DSMILMilitaryMode()

# Search with attestation
results = rag.search("evidence query")

# Generate attested report
attested_report = dsmil.attest_data(results)
# Contains TPM quote verifying data integrity
```

### Multi-Source Intelligence

```python
# Correlate screenshots with chat logs
from ai_analysis_layer import AIAnalysisLayer

ai = AIAnalysisLayer(vector_rag=rag)

# Analyze period of interest
results = ai.analyze_timeline(
    start_time=datetime(2025, 11, 10),
    end_time=datetime(2025, 11, 12),
    auto_detect_incidents=True
)

# Results include:
# - Event links (temporal, content-based)
# - Anomalies (suspicious activity)
# - Patterns (recurring behaviors)
# - Incidents (grouped events)

# Perfect for intelligence analysis
```

---

## Complete Workflow Examples

### Example 1: Comprehensive Intelligence Gathering

```python
from screenshot_intelligence import ScreenshotIntelligence
from telegram_integration import TelegramIntegration, TelegramConfig
from signal_integration import SignalIntegration, SignalConfig
from ai_analysis_layer import AIAnalysisLayer
from datetime import datetime, timedelta

# Initialize all systems
intel = ScreenshotIntelligence()
ai_analysis = AIAnalysisLayer(screenshot_intel=intel)

# 1. Register devices
intel.register_device("phone1", "GrapheneOS Phone", "grapheneos", "/data/phone1/screenshots")
intel.register_device("laptop", "Dell Latitude 5450", "laptop", "/data/laptop/screenshots")

# 2. Ingest all screenshots
intel.scan_device_screenshots("phone1")
intel.scan_device_screenshots("laptop")

# 3. Ingest Telegram messages
telegram_config = TelegramConfig(api_id="...", api_hash="...", phone_number="...")
telegram = TelegramIntegration(telegram_config, vector_rag=intel.rag)
await telegram.start()
await telegram.sync_all_monitored_chats(limit_per_chat=1000)

# 4. Ingest Signal messages
signal_config = SignalConfig(phone_number="...")
signal = SignalIntegration(signal_config, vector_rag=intel.rag)
signal.sync_messages(max_messages=1000)

# 5. Comprehensive analysis
week_ago = datetime.now() - timedelta(days=7)
results = ai_analysis.analyze_timeline(
    start_time=week_ago,
    end_time=datetime.now(),
    auto_detect_incidents=True
)

# 6. Review results
print(f"Total events: {results['analysis']['statistics']['total_events']}")
print(f"Anomalies detected: {len(results['anomalies'])}")
print(f"Patterns found: {len(results['patterns'])}")
print(f"Incidents: {len(results['incidents'])}")

# 7. Generate report
for incident in results['incidents']:
    summary = ai_analysis.generate_incident_summary(incident)
    print(f"\nIncident: {incident.incident_name}")
    print(f"Summary: {summary}")
```

### Example 2: Real-Time Monitoring

```python
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ScreenshotHandler(FileSystemEventHandler):
    def __init__(self, intel):
        self.intel = intel

    def on_created(self, event):
        if event.src_path.endswith('.png'):
            print(f"New screenshot: {event.src_path}")
            result = self.intel.ingest_screenshot(event.src_path, device_id="phone1")
            if result['status'] == 'success':
                print(f"  Ingested with OCR confidence: {result['metadata']['ocr_confidence']:.2%}")

# Monitor screenshot directory
intel = ScreenshotIntelligence()
handler = ScreenshotHandler(intel)
observer = Observer()
observer.schedule(handler, "/data/phone1/screenshots", recursive=False)
observer.start()

# Also monitor chat messages
async def monitor_messages():
    while True:
        # Sync Telegram
        await telegram.sync_all_monitored_chats(incremental=True)

        # Sync Signal
        signal.sync_messages()

        # Wait 5 minutes
        await asyncio.sleep(300)

# Run both monitors
asyncio.run(monitor_messages())
```

### Example 3: CLI Workflow

```bash
# Register devices
screenshot-intel device register phone1 "GrapheneOS Phone" grapheneos /data/phone1/screenshots
screenshot-intel device register laptop "Laptop" laptop /data/laptop/screenshots

# Ingest screenshots
screenshot-intel ingest scan phone1
screenshot-intel ingest scan laptop

# Search
screenshot-intel search "VPN error" --limit 10 --type image

# Timeline
screenshot-intel timeline 2025-11-10 2025-11-12 --format markdown --output timeline.md

# AI Analysis
screenshot-intel analyze 2025-11-10 2025-11-12 --detect-incidents --output analysis.json

# Statistics
screenshot-intel stats
```

### Example 4: API Workflow

```bash
# Start API server
./start-api-server.sh

# Register device (via API)
curl -X POST http://127.0.0.1:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "phone1",
    "device_name": "GrapheneOS Phone",
    "device_type": "grapheneos",
    "screenshot_path": "/data/phone1/screenshots"
  }'

# Scan device
curl -X POST http://127.0.0.1:8000/api/ingest/scan \
  -H "Content-Type: application/json" \
  -d '{"device_id": "phone1", "pattern": "*.png"}'

# Search
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database error", "limit": 10}'

# AI Analysis
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-11-10",
    "end_date": "2025-11-12",
    "detect_incidents": true
  }'
```

---

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU for embeddings and OCR
rag = VectorRAGSystem(use_gpu=True)

# Check if GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {rag.embedding_model.device}")
```

### Batch Processing

```python
# Process screenshots in batches for better performance
from pathlib import Path

screenshots = list(Path("/data/phone1/screenshots").glob("*.png"))

# Process in batches of 32
batch_size = 32
for i in range(0, len(screenshots), batch_size):
    batch = screenshots[i:i+batch_size]
    for screenshot in batch:
        intel.ingest_screenshot(screenshot, device_id="phone1")
    print(f"Processed {i+len(batch)}/{len(screenshots)}")
```

### Caching

```python
# Vector RAG automatically caches embeddings
# To clear cache:
rag.embedding_model._cache = {}
```

---

## Troubleshooting Integration Issues

### Issue: MCP Server Not Found

```bash
# Check MCP config
cat /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json | grep screenshot-intelligence

# Verify path
ls -la /home/user/LAT5150DRVMIL/02-ai-engine/screenshot_intel_mcp_server.py

# Test manually
python3 /home/user/LAT5150DRVMIL/02-ai-engine/screenshot_intel_mcp_server.py
```

### Issue: Qdrant Connection Failed

```bash
# Check if Qdrant is running
curl http://127.0.0.1:6333/collections

# Check Docker
docker ps | grep qdrant

# Restart Qdrant
docker restart qdrant
```

### Issue: OCR Not Working

```python
# Test PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR()
result = ocr.ocr('test.png')
print(result)

# Fallback to Tesseract
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('test.png'))
print(text)
```

---

## Next Steps

1. **Deploy to Production**: Run `./deploy_screenshot_intel_production.sh`
2. **Configure Integrations**: Edit `~/.screenshot_intel/.env` for Telegram/Signal
3. **Register Devices**: Use CLI or API to register screenshot sources
4. **Start Ingesting**: Scan existing screenshots and chat logs
5. **Analyze**: Use AI analysis layer for intelligent insights

---

**Version:** 1.0.0
**Status:** Production Ready
**Platform:** LAT5150DRVMIL - Dell Latitude 5450 Covert Edition
**Integration:** Complete ✅
