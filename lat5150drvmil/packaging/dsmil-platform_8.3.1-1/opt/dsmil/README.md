# DSMIL Unified AI Platform v8.3.1

**Dell Latitude 5450 Covert Edition - LOCAL-FIRST AI with Intelligent Web Crawling**

[![Classification](https://img.shields.io/badge/Classification-JRTC1%20Training-yellow)](https://github.com/SWORDIntel/LAT5150DRVMIL)
[![Mode 5](https://img.shields.io/badge/DSMIL-Mode%205%20STANDARD-green)](./03-security/)
[![TPM](https://img.shields.io/badge/TPM%202.0-Attested-blue)](./02-ai-engine/)
[![Compute](https://img.shields.io/badge/Compute-76.4%20TOPS-red)](./00-documentation/)
[![Web](https://img.shields.io/badge/Web-Crawling%20%2B%20Search-orange)](./04-integrations/)
[![UI](https://img.shields.io/badge/UI-ChatGPT%20Style-brightgreen)](./03-web-interface/)

> Complete LOCAL-FIRST AI development platform with ChatGPT-style interface, intelligent web crawling, auto-coding tools, and DSMIL Mode 5 security. Perfect for offensive security research - no TOS restrictions, 100% private, cryptographically attested.

---

## Quick Start

### Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/SWORDIntel/LAT5150DRVMIL
cd LAT5150DRVMIL

# Run installer (handles everything)
./install.sh

# Access web interface
xdg-open http://localhost:9876
```

**See [INSTALL.md](INSTALL.md) for complete installation guide and troubleshooting.**

### Manual Installation

```bash
# Install dependencies
pip3 install requests anthropic google-generativeai openai flask flask-cors beautifulsoup4 sentence-transformers faiss-cpu

# Install Ollama (for local inference)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:1.5b

# Start the unified server
python3 03-web-interface/dsmil_unified_server.py

# Access web interface
xdg-open http://localhost:9876
```

---

## What Is This?

A complete framework for running local AI inference with **hardware-attested responses** using Dell's DSMIL (Dell System Management Interface Layer) Mode 5 platform integrity features.

### Key Features (v8.3.1 Update)

**NEW in v8.3.1:**
- **🔍 Enhanced Auto-Coding**: Code review, test generation, documentation generation
- **📦 One-Click Install**: Automated installer handles everything (`./install.sh`)
- **🧹 Clean Codebase**: Organized structure with proper documentation
- **📖 Complete Guides**: INSTALL.md, STRUCTURE.md, cleanup scripts

**Features from v8.3:**
- **🤖 Smart Routing**: Auto-detects code vs general queries, routes to specialized models
- **🌐 Web Search**: Integrated DuckDuckGo search for current information
- **🕷️ Web Crawler**: Intelligent site crawling with PDF extraction
- **💻 ChatGPT-Style UI**: Clean 3-panel interface with menu bar and guardrail controls
- **📁 Natural Commands**: "search database for X", "crawl site https://url"
- **🛠️ 7 Auto-Coding Tools**: Edit, Create, Debug, Refactor, Review, Tests, Docs
- **🚀 Launch Claude Code**: One-click launcher in TOOLS menu

**Core Features:**
- **🔒 Hardware Attestation**: Every AI response cryptographically verified via TPM 2.0
- **🚀 76.4 TOPS Compute**: NPU (26.4) + GPU (40) + NCS2 (10)
- **🎯 Multi-Model**: DeepSeek R1 + DeepSeek Coder + Qwen Coder + CodeLlama
- **🛡️ Mode 5 Security**: 84 DSMIL devices for military-grade platform integrity
- **📚 RAG System**: 223 documents, 54M tokens indexed
- **🔐 No Guardrails**: Perfect for offensive security research - local models unrestricted

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified Query Interface                     │
│               (Web UI + REST API + CLI)                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │  GNA Router (<1ms)│  ← Intel GNA 3.0 (always-on)
         └─────────┬─────────┘
                   │
      ┌────────────┼────────────┬────────────┐
      │            │            │            │
┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐ ┌────▼─────┐
│  Local    │ │ Claude │ │  Gemini  │ │  OpenAI  │
│ DeepSeek  │ │  Code  │ │  Flash   │ │  GPT-4   │
│ (Private) │ │  API   │ │   API    │ │   API    │
└─────┬─────┘ └───┬────┘ └────┬─────┘ └────┬─────┘
      │           │            │            │
      │           └────────────┴────────────┘
      │                      │
┌─────▼──────┐        ┌──────▼────────┐
│   DSMIL    │        │  Standard     │
│  Device 16 │        │  JSON         │
│  (Attest)  │        │  Response     │
└────────────┘        └───────────────┘
```

### Hardware Stack

| Component | Performance | Status | Purpose |
|-----------|-------------|--------|---------|
| **Intel NPU 3720** | 26.4 TOPS | ✅ Military Mode | AI acceleration |
| **Intel Arc GPU** | 40 TOPS | ✅ Active | Parallel compute |
| **Intel NCS2** | 10 TOPS | ✅ Detected | Edge inference |
| **Intel GNA 3.0** | 1 GOPS | ✅ Always-on | Command routing |
| **AVX-512** | 12 P-cores | ✅ Unlocked | Vector operations |
| **DSMIL Devices** | 84 devices | ✅ Mode 5 | Platform security |

**Total Compute:** 76.4 TOPS

---

## v8.3 New Features Deep Dive

### Smart Routing System
**Automatic model selection** - Just type naturally, system chooses best model:
- "write a function" → DeepSeek Coder (specialized)
- "what is quantum" → DeepSeek R1 (fast general)
- "latest AI news" → Web search + AI synthesis

### Web Intelligence
**Integrated web research:**
- DuckDuckGo search (privacy-first, no tracking)
- URL scraping: Paste https://url → auto-indexes to RAG
- Site crawling: "scrape all of https://site.com" → crawls entire site
- PDF extraction: Prioritizes PDFs, extracts text automatically
- SSL bypass: Works on research sites with certificate issues

**Example:** Successfully crawled https://www.erowid.org/archive/rhodium/chemistry/ and found 107 chemistry PDFs!

### ChatGPT-Style Interface
**Clean, functional UI:**
- 3-panel layout (sidebar | chat | input)
- Menu bar (FILE, EDIT, TOOLS, RAG, HELP)
- Military green theme (phosphor aesthetic maintained)
- Auto-coding tools in sidebar
- Guardrail controls (temperature, safety, max tokens)
- File browser dialog (no more manual paths!)

### Auto-Coding Capabilities
**Built-in development tools:**
- Edit existing files (Local Claude Code MVP)
- Create new files (AI-generated)
- Debug code (find and fix bugs)
- Refactor code (improve structure)
- Launch Claude Code (one-click from menu)

### Natural Language Commands
**No slash commands required:**
- "search our database for QUANTUM" → RAG search
- "find in knowledge APT techniques" → RAG search
- Paste URL → scrapes and indexes
- "crawl and index all of https://site" → full site crawl

## Directory Structure

```
LAT5150DRVMIL/
├── 00-documentation/          # Comprehensive docs + 3-week redesign plan
├── 01-source/                 # Original DSMIL framework (84 devices)
├── 02-ai-engine/              # AI inference with smart routing + web search
│   ├── smart_router.py        # Auto code detection
│   ├── web_search.py          # DuckDuckGo integration
│   ├── code_specialist.py     # Code generation
│   ├── local_claude_code.py   # Codebase editing (MVP)
│   └── unified_orchestrator.py # Multi-backend coordination
├── 03-web-interface/          # ChatGPT-style UI + server
│   ├── clean_ui_v3.html       # Modern 3-panel interface
│   └── dsmil_unified_server.py # Comprehensive backend
├── 04-integrations/           # RAG, web crawling, tools
│   ├── web_scraper.py         # Intelligent crawler + PDF extraction
│   ├── rag_manager.py         # Knowledge base management
│   └── crawl4ai_wrapper.py    # Industrial crawler (optional)
├── 05-deployment/             # Systemd services, configs
└── 03-security/               # Covert Edition security analysis
```

---

## Usage Examples

See [UNIFIED_PLATFORM_ARCHITECTURE.md](./00-documentation/UNIFIED_PLATFORM_ARCHITECTURE.md) for complete API documentation and sub-agent integration guide.

---

## License

**Classification:** JRTC1 Training Environment
**Distribution:** Educational/Research Use
**Compliance:** DoD 8500 series, NIST Cybersecurity Framework

---

## Credits

Built with hardware-attested AI responses. Every query cryptographically verified via TPM 2.0.

🔒 **Mode 5 STANDARD** | 🎯 **76.4 TOPS** | ⚡ **AVX-512 Unlocked**
