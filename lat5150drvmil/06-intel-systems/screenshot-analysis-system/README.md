# Screenshot Intelligence System

**AI-Driven Screenshot Organization & Analysis**

> Transform scattered screenshots and chat logs into a structured, searchable intelligence repository with AI-powered timeline analysis and event correlation.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-Production-green)
![Integration](https://img.shields.io/badge/integration-LAT5150DRVMIL-orange)

---

## 🎯 Overview

The Screenshot Intelligence System is an AI-driven platform that transforms unorganized screenshots and chat logs from multiple devices into a structured, searchable knowledge base. Built on top of the existing LAT5150DRVMIL RAG system, it provides semantic search, timeline reconstruction, and automated event correlation.

### Key Features

✅ **Screenshot Ingestion & OCR**
- Automatic text extraction from screenshots (PaddleOCR + Tesseract)
- Metadata extraction (timestamps, device info)
- Multi-device support (GrapheneOS, laptops, desktops)

✅ **Vector Database (Qdrant)**
- Upgraded from TF-IDF (51.8% accuracy) to transformer embeddings (88%+ target)
- Semantic search with BAAI/bge-base-en-v1.5 model
- 384-dimensional vector embeddings

✅ **Chat Log Integration**
- Telegram message timeline (uses existing `telegram_document_scraper.py`)
- Signal integration via signal-cli
- Cross-platform message correlation

✅ **Timeline Analysis**
- Chronological event reconstruction
- Time-based correlation (configurable windows)
- Incident grouping and clustering

✅ **AI-Powered Analysis**
- Event linking and relationship discovery
- Anomaly detection
- Automated summarization
- Content classification

✅ **Security & Privacy**
- LOCAL-FIRST: All processing on-device
- Services bound to 127.0.0.1 (no external access)
- TPM 2.0 hardware attestation compatible
- SWORD Intelligence integration ready

✅ **Production Features** (NEW)
- Comprehensive health monitoring and diagnostics
- Automated maintenance tasks (log rotation, cleanup, optimization)
- Retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Graceful degradation and fallback mechanisms
- Performance metrics collection
- Automated backup and recovery
- Integration test suite
- Production deployment automation

---

## 📊 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Screenshot Intelligence                   │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Screenshots │  │ Chat Logs    │  │ System Logs     │   │
│  │ (OCR)       │  │ (Telegram/   │  │ (Optional)      │   │
│  │             │  │  Signal)     │  │                 │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘   │
│         │                 │                    │             │
│         └─────────────────┴────────────────────┘             │
│                           │                                  │
│                ┌──────────▼──────────┐                      │
│                │  Vector RAG System  │                      │
│                │  (Qdrant + BAAI)    │                      │
│                └──────────┬──────────┘                      │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐               │
│         │                 │                 │               │
│  ┌──────▼────┐  ┌────────▼────────┐  ┌────▼─────┐         │
│  │ Semantic  │  │ Timeline        │  │ Event    │         │
│  │ Search    │  │ Queries         │  │ Linking  │         │
│  └───────────┘  └─────────────────┘  └──────────┘         │
│                                                              │
└───────────────────────┬──────────────────────────────────── │
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼────┐              ┌────────▼────────┐
    │ MCP     │              │ DSMIL AI Engine │
    │ Server  │              │ (Orchestrator)  │
    └─────────┘              └─────────────────┘
```

### Components

**1. Vector RAG System** (`04-integrations/rag_system/vector_rag_system.py`)
- Qdrant vector database
- Sentence-transformers embeddings
- Hybrid search (semantic + metadata filtering)

**2. Screenshot Intelligence** (`04-integrations/rag_system/screenshot_intelligence.py`)
- Device management
- Screenshot ingestion with OCR
- Timeline reconstruction
- Event correlation

**3. MCP Server** (`02-ai-engine/screenshot_intel_mcp_server.py`)
- Model Context Protocol integration
- 7 tools for screenshot intelligence
- Secure local-only access

**4. Existing Integrations**
- `telegram_document_scraper.py` - Telegram message scraping
- `telegram_cve_scraper.py` - CVE intelligence from Telegram
- `donut_pdf_processor.py` - Advanced PDF OCR
- `osint_comprehensive.py` - OSINT framework

---

## 🚀 Quick Start

### Prerequisites

- **OS**: Linux (Debian/Ubuntu recommended)
- **Python**: 3.10+
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 20 GB for vector DB and data
- **Docker**: For Qdrant (optional, but recommended)

### Installation

**1. Run Setup Script**

```bash
cd /home/user/LAT5150DRVMIL/06-intel-systems/screenshot-analysis-system
./setup_screenshot_intel.sh
```

This installs:
- System dependencies (Tesseract OCR, etc.)
- Python packages (Qdrant, sentence-transformers, PaddleOCR)
- Qdrant vector database (Docker)
- MCP server configuration

**2. Verify Installation**

```bash
# Check Qdrant is running
curl http://127.0.0.1:6333/collections

# Test Vector RAG
python3 04-integrations/rag_system/vector_rag_system.py

# Test Screenshot Intelligence
python3 04-integrations/rag_system/screenshot_intelligence.py
```

**3. Register Devices**

```python
from screenshot_intelligence import ScreenshotIntelligence

intel = ScreenshotIntelligence()

# Register GrapheneOS phone
intel.register_device(
    device_id="phone1",
    device_name="GrapheneOS Phone 1",
    device_type="grapheneos",
    screenshot_path="/path/to/phone1/screenshots"
)

# Register laptop
intel.register_device(
    device_id="laptop",
    device_name="Dell Latitude 5450",
    device_type="laptop",
    screenshot_path="/path/to/laptop/screenshots"
)
```

---

## 📚 Usage

### 1. Screenshot Ingestion

**Ingest Single Screenshot:**

```python
from screenshot_intelligence import ScreenshotIntelligence

intel = ScreenshotIntelligence()

result = intel.ingest_screenshot(
    screenshot_path="/path/to/Screenshot_20251112_143022.png",
    device_id="phone1"
)

print(result)
# {
#   'status': 'success',
#   'id': 'abc123...',
#   'type': 'image',
#   'metadata': {
#     'ocr_engine': 'paddleocr',
#     'ocr_confidence': 0.94,
#     'timestamp': '2025-11-12T14:30:22',
#     'device_name': 'GrapheneOS Phone 1'
#   }
# }
```

**Scan Entire Device:**

```python
result = intel.scan_device_screenshots(
    device_id="phone1",
    pattern="*.png"
)

print(f"Ingested: {result['success']}")
print(f"Already indexed: {result['already_indexed']}")
print(f"Errors: {result['errors']}")
```

### 2. Semantic Search

**Search Screenshots & Chat Logs:**

```python
results = intel.rag.search(
    query="VPN connection error",
    limit=10,
    score_threshold=0.6
)

for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Type: {result.document.doc_type}")
    print(f"Timestamp: {result.document.timestamp}")
    print(f"Text: {result.document.text[:200]}")
    print()
```

**Search with Filters:**

```python
from datetime import datetime

results = intel.rag.search(
    query="error message",
    limit=5,
    filters={
        'type': 'image',  # Only screenshots
        'source': 'screenshot',
        'date_from': datetime(2025, 11, 1),
        'date_to': datetime(2025, 11, 12)
    }
)
```

### 3. Timeline Queries

**Query by Date Range:**

```python
from datetime import datetime

events = intel.rag.timeline_query(
    start_time=datetime(2025, 11, 10, 0, 0),
    end_time=datetime(2025, 11, 12, 23, 59),
    doc_types=['image', 'chat_message']
)

for event in events:
    print(f"[{event.timestamp}] {event.doc_type}: {event.filename}")
```

**Generate Timeline Report:**

```python
report = intel.generate_timeline_report(
    start_time=datetime(2025, 11, 10),
    end_time=datetime(2025, 11, 12),
    output_format='markdown'
)

print(report)
# Saves to file or displays
```

### 4. Event Correlation

**Find Related Events:**

```python
from screenshot_intelligence import Event

# Create event from screenshot
event = Event(
    event_id="abc123",
    event_type="screenshot",
    timestamp=datetime(2025, 11, 12, 14, 30),
    content="VPN connection failed error 1234"
)

# Find related events (within ±30 min)
related = intel.find_related_events(
    event,
    time_window_before=600,   # 10 minutes before
    time_window_after=1800,   # 30 minutes after
    similarity_threshold=0.6
)

for related_event, score, relation_type in related:
    print(f"Score: {score:.2f} | Type: {relation_type}")
    print(f"Event: {related_event.event_type} at {related_event.timestamp}")
```

### 5. Incident Management

**Create Incident:**

```python
incident = intel.create_incident(
    incident_name="VPN Outage - 2025-11-12",
    event_ids=["abc123", "def456", "ghi789"],
    tags=["network", "vpn", "outage"]
)

print(f"Incident created: {incident.incident_id}")
```

### 6. MCP Server Usage

**Via Claude Code or MCP Client:**

```json
{
  "tool": "search_intel",
  "arguments": {
    "query": "database error",
    "limit": 5,
    "doc_type": "image"
  }
}
```

**Available MCP Tools:**
1. `ingest_screenshot` - Ingest single screenshot
2. `scan_device` - Scan device directory
3. `search_intel` - Semantic search
4. `timeline_query` - Query by date range
5. `generate_timeline_report` - Generate timeline report
6. `register_device` - Register new device
7. `get_stats` - Get system statistics

---

## 🔧 Configuration

### Data Directories

Default location: `~/.screenshot_intel/`

```
~/.screenshot_intel/
├── screenshots/
│   ├── phone1/
│   ├── phone2/
│   └── laptop/
├── chat_logs/
│   ├── telegram/
│   └── signal/
├── incidents/
│   └── *.json
├── logs/
└── devices.json
```

### Qdrant Configuration

**Local Docker (recommended):**
```bash
docker run -d \
  --name qdrant \
  -p 127.0.0.1:6333:6333 \
  -v $HOME/qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant
```

**Python Configuration:**
```python
from vector_rag_system import VectorRAGSystem

rag = VectorRAGSystem(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="lat5150_knowledge_base",
    embedding_model="BAAI/bge-base-en-v1.5",
    use_gpu=True
)
```

### OCR Engines

**PaddleOCR (default):**
- Best accuracy for screenshots
- GPU acceleration supported
- Multi-language support

**Tesseract (fallback):**
- Lightweight
- Good for text-heavy images

Configure in `vector_rag_system.py`:
```python
# PaddleOCR settings
self.paddle_ocr = PaddleOCR(
    use_angle_cls=True,  # Handle rotated text
    lang='en',           # English
    use_gpu=True,        # GPU acceleration
    show_log=False
)
```

---

## 🔐 Security

### Local-Only Access

All services bound to `127.0.0.1` (localhost):
- Qdrant: `127.0.0.1:6333`
- MCP servers: stdio (no network binding)
- No external network access

### Data Privacy

- **LOCAL-FIRST**: All processing on-device
- **No cloud dependencies**: Ollama + local models
- **Encryption at rest**: Optional (via ZFS encryption)
- **TPM attestation**: Compatible with DSMIL hardware security

### MCP Server Security

- stdio transport (no network exposure)
- No authentication required (local-only)
- Sandboxed execution environment

---

## 📈 Performance

### Accuracy Metrics

**Before (TF-IDF):**
- Average accuracy: 51.8%
- Search method: Keyword matching

**After (Vector DB):**
- Target accuracy: 88%+ (based on research)
- Search method: Semantic embeddings
- Model: BAAI/bge-base-en-v1.5 (384D)

### Resource Usage

**RAM:**
- Base system: ~2 GB
- Qdrant: ~1 GB
- Embedding model: ~500 MB
- OCR (PaddleOCR): ~1 GB
- **Total: ~4.5 GB**

**Storage:**
- Vector DB: ~10 MB per 1000 screenshots
- Metadata: ~1 MB per 1000 items
- Screenshots (original): Variable

**Compute:**
- NPU acceleration: Supported (Intel AI Boost)
- GPU acceleration: Supported (CUDA/ROCm)
- CPU fallback: Yes

---

## 🧩 Integration

### With Existing Systems

**1. Telegram Scrapers**

```python
from telegram_document_scraper import TelegramDocumentScraper
from vector_rag_system import VectorRAGSystem

# Scrape Telegram messages
scraper = TelegramDocumentScraper()
messages = scraper.get_messages(chat_id="...", limit=100)

# Ingest into vector DB
rag = VectorRAGSystem()
for msg in messages:
    rag.ingest_chat_message(
        message=msg['text'],
        source='telegram',
        chat_id=msg['chat_id'],
        chat_name=msg['chat_name'],
        sender=msg['sender'],
        timestamp=msg['timestamp']
    )
```

**2. DSMIL AI Engine**

```python
from dsmil_ai_engine import DSMILAIEngine
from screenshot_intelligence import ScreenshotIntelligence

# Query screenshot intelligence
intel = ScreenshotIntelligence()
results = intel.rag.search("database error in logs", limit=3)

# Generate summary with AI
ai = DSMILAIEngine()
context = '\n'.join([r.document.text for r in results])
prompt = f"Summarize these error logs:\n\n{context}"
summary = ai.query(prompt, model="quality_code")

print(summary)
```

**3. SWORD Intelligence**

- Post-quantum crypto compatible
- Hardware attestation ready
- Forensic timeline analysis
- Evidence preservation

---

## 📝 Advanced Usage

### Custom Embedding Models

```python
# Use different embedding model
rag = VectorRAGSystem(
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    use_gpu=True
)
```

### Batch Processing

```python
from pathlib import Path

screenshots = list(Path("/screenshots").glob("*.png"))

for screenshot in screenshots:
    try:
        result = intel.ingest_screenshot(screenshot, device_id="phone1")
        if result['status'] == 'success':
            print(f"✓ {screenshot.name}")
    except Exception as e:
        print(f"✗ {screenshot.name}: {e}")
```

### Export Timeline

```python
import json

# Get events
events = intel.rag.timeline_query(
    start_time=datetime(2025, 11, 1),
    end_time=datetime(2025, 11, 30)
)

# Export to JSON
timeline_data = [{
    'timestamp': e.timestamp.isoformat(),
    'type': e.doc_type,
    'text': e.text,
    'metadata': e.metadata
} for e in events]

with open('timeline_export.json', 'w') as f:
    json.dump(timeline_data, f, indent=2)
```

---

## 🐛 Troubleshooting

### Qdrant Connection Failed

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check logs
docker logs qdrant

# Restart Qdrant
docker restart qdrant
```

### OCR Not Working

```bash
# Test Tesseract
tesseract --version

# Test PaddleOCR
python3 -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(); print('OK')"

# Check GPU availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Low Search Accuracy

1. **Increase embedding quality:**
   - Use larger model: `sentence-transformers/all-mpnet-base-v2`
   - Enable GPU acceleration

2. **Tune search parameters:**
   ```python
   results = rag.search(
       query="...",
       score_threshold=0.5,  # Lower = more results
       limit=20              # More candidates
   )
   ```

3. **Improve OCR quality:**
   - Pre-process images (contrast, grayscale)
   - Use PaddleOCR over Tesseract

---

## 📖 API Reference

See individual module documentation:

- **Vector RAG System**: `04-integrations/rag_system/vector_rag_system.py`
- **Screenshot Intelligence**: `04-integrations/rag_system/screenshot_intelligence.py`
- **MCP Server**: `02-ai-engine/screenshot_intel_mcp_server.py`

---

## 🤝 Contributing

This is an internal LAT5150DRVMIL subsystem. Enhancements should:
- Maintain security (local-only)
- Follow existing architecture
- Document API changes
- Test with real data

---

## 📜 License

Internal LAT5150DRVMIL project. See main repository license.

---

## 📞 Support

**Documentation:**
- Main README: `/home/user/LAT5150DRVMIL/README.md`
- RAG System: `04-integrations/rag_system/README.md`
- SWORD Intelligence: `00-documentation/00-root-docs/SWORD_INTELLIGENCE.md`
- **Production Best Practices**: `06-intel-systems/PRODUCTION_BEST_PRACTICES.md` ⭐ NEW
- **Integration Guide**: `06-intel-systems/INTEGRATION_GUIDE.md`
- **Deployment Guide**: `06-intel-systems/SCREENSHOT_INTEL_DEPLOYMENT.md`

**Health & Monitoring:**
```bash
# Run health check
~/.screenshot_intel/run-health-check.sh

# Run maintenance
~/.screenshot_intel/run-maintenance.sh

# Collect metrics
python3 04-integrations/rag_system/system_health_monitor.py --metrics
```

**Testing:**
```bash
# Unit/integration tests
python3 04-integrations/rag_system/test_screenshot_intel_integration.py -v

# Component tests
python3 04-integrations/rag_system/vector_rag_system.py
python3 04-integrations/rag_system/screenshot_intelligence.py
```

---

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-11-12
**Platform:** LAT5150DRVMIL - Dell Latitude 5450 Covert Edition
