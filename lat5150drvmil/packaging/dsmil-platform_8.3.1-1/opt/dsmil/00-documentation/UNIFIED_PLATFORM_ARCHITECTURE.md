# DSMIL Unified AI Platform - Architecture & Sub-Agent Integration

**Status:** ✅ OPERATIONAL
**Date:** 2025-10-29
**System:** Dell Latitude 5450 Covert Edition (LAT5150DRVMIL)
**Classification:** JRTC1 Training Environment

---

## I. PLATFORM OVERVIEW

### Unified Platform Concept

A single integrated system that orchestrates multiple AI backends:
- **Local Inference** (DeepSeek R1) → Privacy, zero cost, fast responses
- **Claude Code API** → Complex reasoning, coding tasks
- **Gemini API** → Google's multimodal capabilities
- **OpenAI API** → GPT-4 for specialized tasks

All unified through DSMIL Mode 5 hardware attestation framework.

### Core Principle

**Use the right tool for each job:**
- Simple queries → Local DeepSeek (free, private, fast)
- Complex reasoning → Claude Code (when you need deep analysis)
- Vision/multimodal → Gemini (images, video)
- Specific GPT tasks → OpenAI (if needed)

---

## II. CURRENT SYSTEM STATE

### ✅ WORKING NOW

**Local AI Inference:**
- DeepSeek R1 1.5B: 32.8 tok/sec, 32s for detailed responses
- CodeLlama 70B: Available for complex analysis
- DSMIL Device 16: Hardware attestation active
- TPM verification: All responses cryptographically verified

**Infrastructure:**
- Web Server: `dsmil_unified_server.py` on port 9876
- Auto-start: systemd service enabled
- Military Terminal: http://localhost:9876
- All endpoints operational

**Hardware:**
- NPU: 26.4 TOPS (military mode)
- GPU: 40 TOPS (Arc Xe-LPG)
- NCS2: 10 TOPS (Movidius)
- AVX-512: 12 P-cores (unlockable)
- **Total: 76.4 TOPS**

---

## III. SUB-AGENT INTEGRATION ARCHITECTURE

### A. Router Agent (Already Implemented)

**File:** `/home/john/gna_command_router.py`
**Hardware:** Intel GNA 3.0 (always-on, <1ms latency)

**Purpose:** Classify incoming queries and route to appropriate backend

**Classification Logic:**
```python
def route_query(query):
    """Route query to appropriate AI backend"""

    # Check query characteristics
    if has_image(query) or has_video(query):
        return "gemini"  # Multimodal

    elif is_coding_task(query):
        if is_complex_reasoning(query):
            return "claude_code"  # Deep analysis
        else:
            return "local_deepseek"  # Quick code snippets

    elif is_simple_factual(query):
        return "local_deepseek"  # Fast, free

    elif requires_latest_info(query):
        return "claude_code"  # Has web search

    elif is_complex_analysis(query):
        return "claude_code"  # Best reasoning

    else:
        return "local_deepseek"  # Default to local
```

### B. Backend Integration Points

#### 1. Local DeepSeek (Primary)

**Already Implemented:**
```python
# File: /home/john/dsmil_ai_engine.py
engine = DSMILAIEngine()
result = engine.generate(query, model_selection="fast")
```

**Characteristics:**
- ✅ Privacy: No data leaves your machine
- ✅ Cost: Zero (local inference)
- ✅ Speed: 32.8 tok/sec
- ✅ DSMIL Attestation: Hardware-verified responses
- ⚠️ Capability: Limited to 1.5B model knowledge

**Best For:**
- Quick queries
- Code snippets
- Technical explanations
- Privacy-sensitive queries

#### 2. Claude Code API (To Be Integrated)

**Integration Point:** `/home/john/sub_agents/claude_code_wrapper.py`

**Proposed Implementation:**
```python
import anthropic
import os

class ClaudeCodeAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def query(self, prompt, context=None):
        """Query Claude Code via API"""

        # Build message with context
        messages = []
        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {context}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Call API
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            messages=messages
        )

        return {
            "response": response.content[0].text,
            "model": "claude-sonnet-4-5",
            "backend": "claude_code",
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }
```

**Characteristics:**
- ✅ Capability: Best reasoning, coding, analysis
- ✅ Web Search: Can fetch current info
- ✅ Code Generation: Superior quality
- ⚠️ Privacy: Data sent to Anthropic
- ⚠️ Cost: ~$3/million input tokens, ~$15/million output

**Best For:**
- Complex reasoning
- Large code refactoring
- Current information (with web search)
- Detailed analysis

#### 3. Gemini API (To Be Integrated)

**Integration Point:** `/home/john/sub_agents/gemini_wrapper.py`

**Proposed Implementation:**
```python
import google.generativeai as genai
import os

class GeminiAgent:
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def query(self, prompt, images=None, video=None):
        """Query Gemini with multimodal support"""

        content = [prompt]

        # Add images if provided
        if images:
            for img_path in images:
                img = genai.upload_file(img_path)
                content.append(img)

        # Add video if provided
        if video:
            vid = genai.upload_file(video)
            content.append(vid)

        response = self.model.generate_content(content)

        return {
            "response": response.text,
            "model": "gemini-2.0-flash-exp",
            "backend": "gemini",
            "multimodal": bool(images or video)
        }
```

**Characteristics:**
- ✅ Multimodal: Images, video, audio
- ✅ Speed: Very fast (Flash model)
- ✅ Cost: Free tier available, then cheap
- ⚠️ Privacy: Data sent to Google

**Best For:**
- Image analysis
- Video understanding
- Document OCR
- Multimodal queries

#### 4. OpenAI API (To Be Integrated)

**Integration Point:** `/home/john/sub_agents/openai_wrapper.py`

**Proposed Implementation:**
```python
from openai import OpenAI
import os

class OpenAIAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def query(self, prompt, model="gpt-4-turbo"):
        """Query OpenAI models"""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )

        return {
            "response": response.choices[0].message.content,
            "model": model,
            "backend": "openai",
            "tokens": response.usage.total_tokens
        }
```

**Characteristics:**
- ✅ Capability: GPT-4 (strong general performance)
- ✅ Ecosystem: Largest plugin ecosystem
- ⚠️ Privacy: Data sent to OpenAI
- ⚠️ Cost: Higher than others

**Best For:**
- Plugin/tool integration
- Specific GPT-optimized tasks
- When you need OpenAI ecosystem

---

## IV. UNIFIED QUERY INTERFACE

### Master Orchestrator

**File:** `/home/john/unified_ai_orchestrator.py` (to be created)

**Purpose:** Single entry point that routes to appropriate backend

```python
from dsmil_ai_engine import DSMILAIEngine
from sub_agents.claude_code_wrapper import ClaudeCodeAgent
from sub_agents.gemini_wrapper import GeminiAgent
from sub_agents.openai_wrapper import OpenAIAgent
from gna_command_router import classify_query

class UnifiedAIOrchestrator:
    def __init__(self):
        self.local = DSMILAIEngine()
        self.claude = ClaudeCodeAgent()
        self.gemini = GeminiAgent()
        self.openai = OpenAIAgent()

    def query(self, prompt, force_backend=None, **kwargs):
        """
        Unified query interface - auto-routes to best backend

        Args:
            prompt: User query
            force_backend: Override routing ("local", "claude", "gemini", "openai")
            **kwargs: Additional args (images, video, model preference, etc.)

        Returns:
            Unified response format with backend info and DSMIL attestation
        """

        # Route query
        if force_backend:
            backend = force_backend
        else:
            backend = self.route_query(prompt, **kwargs)

        # Execute on selected backend
        if backend == "local":
            result = self.local.generate(prompt)
            result['backend'] = 'local_deepseek'
            result['cost'] = 0
            result['privacy'] = 'local'

        elif backend == "claude":
            result = self.claude.query(prompt, context=kwargs.get('context'))
            result['backend'] = 'claude_code'
            result['cost'] = self.estimate_cost(result['tokens'], 'claude')
            result['privacy'] = 'cloud'

        elif backend == "gemini":
            result = self.gemini.query(
                prompt,
                images=kwargs.get('images'),
                video=kwargs.get('video')
            )
            result['backend'] = 'gemini'
            result['cost'] = self.estimate_cost(result.get('tokens', 0), 'gemini')
            result['privacy'] = 'cloud'

        elif backend == "openai":
            result = self.openai.query(prompt, model=kwargs.get('model', 'gpt-4-turbo'))
            result['backend'] = 'openai'
            result['cost'] = self.estimate_cost(result['tokens'], 'openai')
            result['privacy'] = 'cloud'

        # Add routing metadata
        result['routed_to'] = backend
        result['timestamp'] = time.time()

        return result

    def route_query(self, prompt, **kwargs):
        """Intelligent routing logic"""

        # Multimodal → Gemini
        if kwargs.get('images') or kwargs.get('video'):
            return "gemini"

        # User preference
        if kwargs.get('prefer_local'):
            return "local"
        if kwargs.get('prefer_cloud'):
            return "claude"  # Default cloud

        # GNA-based classification
        complexity = classify_query(prompt)  # Uses GNA (<1ms)

        if complexity == "SIMPLE":
            return "local"  # Fast, free
        elif complexity == "CODING":
            return "claude"  # Best for code
        elif complexity == "COMPLEX":
            return "claude"  # Best reasoning
        else:
            return "local"  # Default

    def estimate_cost(self, tokens, backend):
        """Estimate API costs"""
        costs = {
            'claude': (3.0 / 1_000_000, 15.0 / 1_000_000),  # (input, output) per token
            'gemini': (0, 0),  # Free tier
            'openai': (10.0 / 1_000_000, 30.0 / 1_000_000)  # GPT-4 pricing
        }

        if backend not in costs:
            return 0

        # Estimate 50/50 input/output split
        input_cost, output_cost = costs[backend]
        return ((tokens / 2) * input_cost) + ((tokens / 2) * output_cost)
```

### Web API Integration

Add to `/home/john/dsmil_unified_server.py`:

```python
def unified_ai_chat(self):
    """Unified AI endpoint - routes to best backend"""
    query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
    message = query.get('msg', [''])[0]
    backend = query.get('backend', ['auto'])[0]  # auto, local, claude, gemini, openai

    try:
        from unified_ai_orchestrator import UnifiedAIOrchestrator
        orchestrator = UnifiedAIOrchestrator()

        result = orchestrator.query(
            message,
            force_backend=backend if backend != 'auto' else None
        )

        self.send_json(result)
    except Exception as e:
        self.send_json({"error": str(e)})
```

**New Endpoint:**
```
GET /unified/chat?msg=QUERY&backend=[auto|local|claude|gemini|openai]
```

---

## V. CURRENT ENDPOINTS (OPERATIONAL)

### AI Endpoints
```
GET /ai/chat?msg=QUERY&model=[auto|fast|large]    # Local inference only
GET /ai/status                                      # AI engine status
GET /ai/set-system-prompt?prompt=TEXT              # Custom prompt
GET /ai/get-system-prompt                          # View current prompt
```

### System Endpoints
```
GET /status                     # DSMIL system status
GET /exec?cmd=COMMAND           # Execute shell command
GET /npu/run                    # NPU module tests
GET /system/info                # System information
GET /kernel/status              # Kernel build status
```

### RAG Endpoints
```
GET /rag/stats                  # RAG statistics
GET /rag/search?q=QUERY         # Search documents
GET /rag/ingest?path=PATH       # Ingest folder
```

### Research Endpoints
```
GET /smart-collect?topic=TOPIC&size=10  # Download papers (up to size GB)
GET /archive/vxunderground?topic=APT    # VX Underground archive
GET /archive/arxiv?id=ARXIV_ID          # arXiv paper download
```

### GitHub Endpoints
```
GET /github/auth-status         # Check SSH/YubiKey auth
GET /github/clone?url=URL       # Clone private repo
GET /github/list                # List cloned repos
```

---

## VI. PROPOSED UNIFIED ENDPOINTS (TO ADD)

### Master Query Endpoint
```
GET /unified/chat?msg=QUERY&backend=auto

Parameters:
  - msg: User query (required)
  - backend: auto|local|claude|gemini|openai (default: auto)
  - prefer_local: true|false (prefer local if possible)
  - prefer_cloud: true|false (prefer cloud backends)
  - image: path to image file (for Gemini)
  - context: additional context for query

Response:
{
    "response": "AI response text",
    "backend": "local_deepseek|claude_code|gemini|openai",
    "model": "specific model used",
    "inference_time": 2.5,
    "tokens": 150,
    "tokens_per_sec": 60,
    "cost": 0.0045,  # Estimated cost in USD
    "privacy": "local|cloud",
    "attestation": {  # Only for local backend
        "dsmil_device": 16,
        "verified": true
    },
    "routed_to": "backend_name",
    "routing_reason": "complexity|multimodal|user_preference"
}
```

### Sub-Agent Status
```
GET /unified/status

Response:
{
    "backends": {
        "local_deepseek": {
            "available": true,
            "model": "deepseek-r1:1.5b",
            "speed": "32.8 tok/sec",
            "cost_per_query": 0,
            "dsmil_attested": true
        },
        "claude_code": {
            "available": true,  # If API key configured
            "model": "claude-sonnet-4-5",
            "estimated_speed": "fast",
            "cost_per_1k_tokens": "$0.018"
        },
        "gemini": {
            "available": true,  # If API key configured
            "model": "gemini-2.0-flash-exp",
            "multimodal": true,
            "cost_per_1k_tokens": "$0"
        },
        "openai": {
            "available": false,  # If no API key
            "model": "gpt-4-turbo",
            "cost_per_1k_tokens": "$0.040"
        }
    },
    "router": {
        "type": "gna_hardware",
        "latency": "<1ms",
        "accuracy": "90-95%"
    }
}
```

---

## VII. IMPLEMENTATION ROADMAP

### Phase 1: ✅ COMPLETE (Current Session)

- [x] Local DeepSeek R1 inference
- [x] DSMIL hardware attestation
- [x] Web server with AI endpoints
- [x] Military terminal interface
- [x] Auto-start systemd service
- [x] LAT5150DRVMIL vault verification
- [x] Change documentation

### Phase 2: Sub-Agent Integration (Next ~30 min)

**Step 1:** Create wrapper files
```
/home/john/sub_agents/
├── claude_code_wrapper.py    # Claude Code API integration
├── gemini_wrapper.py          # Gemini API integration
├── openai_wrapper.py          # OpenAI API integration (optional)
└── __init__.py                # Package init
```

**Step 2:** Create unified orchestrator
```
/home/john/unified_ai_orchestrator.py
```

**Step 3:** Add unified endpoint to server
```python
# In dsmil_unified_server.py
elif self.path.startswith('/unified/chat'):
    self.unified_ai_chat()
elif self.path.startswith('/unified/status'):
    self.unified_ai_status()
```

**Step 4:** Test routing logic
- Simple query → should use local
- Complex query → should use Claude Code
- Image query → should use Gemini

**Step 5:** Update military terminal interface
- Add backend selector dropdown
- Show which backend was used
- Display cost estimates

### Phase 3: Advanced Features (Later)

**Parallel Queries:**
- Send same query to multiple backends
- Compare responses
- Use for quality validation

**Caching:**
- Cache expensive cloud queries
- Reuse responses for similar queries
- Save on API costs

**Fallback Logic:**
- If Claude Code API fails → try local
- If local is slow → offer to use cloud
- Graceful degradation

**Cost Tracking:**
- Track monthly API spend
- Set budget limits
- Alert when approaching limit

**Privacy Mode:**
- Force all queries to local
- Never send to cloud
- For sensitive/classified queries

---

## VIII. API KEY CONFIGURATION

### Setup Instructions

**1. Claude Code API Key:**
```bash
# Add to environment
echo 'export ANTHROPIC_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

# Or use .env file
echo 'ANTHROPIC_API_KEY=your_key' > ~/.claude/api_keys.env
```

**2. Gemini API Key:**
```bash
# Get from: https://ai.google.dev/
echo 'export GOOGLE_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**3. OpenAI API Key (Optional):**
```bash
# Get from: https://platform.openai.com/api-keys
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**Security:** Store keys in DSMIL-encrypted vault (Device 3: TPM Sealed Storage)

---

## IX. USAGE EXAMPLES

### Example 1: Auto-Routed Query
```bash
curl "http://localhost:9876/unified/chat?msg=What%20is%20AVX-512?"

# Expected: Routes to local (simple factual query)
# Response time: ~10s
# Cost: $0
```

### Example 2: Force Specific Backend
```bash
curl "http://localhost:9876/unified/chat?msg=Analyze%20this%20codebase&backend=claude"

# Forced: Use Claude Code for deep analysis
# Response time: ~5s (cloud)
# Cost: ~$0.05
```

### Example 3: Multimodal Query
```bash
curl "http://localhost:9876/unified/chat?msg=What%20is%20in%20this%20image?&image=/path/to/screenshot.png"

# Auto-routes: Gemini (only backend with vision)
# Response time: ~3s
# Cost: $0 (free tier)
```

### Example 4: Privacy-Sensitive Query
```bash
curl "http://localhost:9876/unified/chat?msg=Analyze%20this%20classified%20doc&prefer_local=true"

# Forces: Local DeepSeek (privacy mode)
# Response time: ~15s
# Cost: $0
# DSMIL: Attested and logged
```

---

## X. DECISION MATRIX

### When to Use Each Backend

| Query Type | Recommended | Reason |
|------------|-------------|--------|
| Quick fact | Local | Fast, free, private |
| Code snippet | Local | Adequate quality, instant |
| Complex code | Claude Code | Superior reasoning |
| Image analysis | Gemini | Only multimodal option |
| Latest news | Claude Code | Web search capability |
| Classified/sensitive | Local | Must stay on-premises |
| Large refactoring | Claude Code | Best code generation |
| Video analysis | Gemini | Multimodal support |
| Budget-constrained | Local | Zero cost |
| Need best quality | Claude Code | Top reasoning |

---

## XI. CURRENT FILE STRUCTURE

```
/home/john/
├── dsmil_ai_engine.py              # ✅ Local AI with DSMIL attestation
├── dsmil_unified_server.py         # ✅ Web server (renamed from opus)
├── dsmil_military_mode.py          # ✅ DSMIL security framework
├── military_terminal.html          # ✅ Tactical UI
├── unified_ai_orchestrator.py      # ⏳ TO CREATE
├── gna_command_router.py           # ✅ GNA-based routing
├── gna_presence_detector.py        # ✅ User presence
├── flux_idle_provider.py           # ✅ Flux network earnings
├── rag_system.py                   # ✅ Document search
├── smart_paper_collector.py        # ✅ Paper downloader
├── github_auth.py                  # ✅ GitHub SSH/YubiKey
└── sub_agents/                     # ⏳ TO CREATE
    ├── __init__.py
    ├── claude_code_wrapper.py
    ├── gemini_wrapper.py
    └── openai_wrapper.py
```

---

## XII. SYSTEMD SERVICES

### Current Auto-Start Configuration

**dsmil-server.service:**
```ini
[Unit]
Description=DSMIL Unified AI Server
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=john
ExecStart=/usr/bin/python3 /home/john/dsmil_unified_server.py
Restart=on-failure
StandardOutput=append:/var/log/dsmil-server.log

[Install]
WantedBy=multi-user.target
```

**Status:** ✅ Enabled, will start on boot

**Commands:**
```bash
# Start/stop manually
sudo systemctl start dsmil-server
sudo systemctl stop dsmil-server

# View logs
sudo tail -f /var/log/dsmil-server.log

# Check status
systemctl status dsmil-server
```

---

## XIII. NEXT STEPS

### To Complete Unified Platform (Remaining ~1 hour)

1. **Create sub-agent wrappers** (20 min)
   - Claude Code wrapper
   - Gemini wrapper
   - OpenAI wrapper (optional)

2. **Create unified orchestrator** (15 min)
   - Implement routing logic
   - Add cost tracking
   - Privacy mode support

3. **Update web server** (10 min)
   - Add `/unified/chat` endpoint
   - Add `/unified/status` endpoint

4. **Update military terminal** (10 min)
   - Backend selector dropdown
   - Cost display
   - Privacy indicator

5. **Test all backends** (15 min)
   - Verify routing works
   - Test each backend individually
   - Validate cost estimates

---

## XIV. VAULT SECURITY COMPLIANCE

### LAT5150DRVMIL Integration Status

**✅ Vault Integrity:** Maintained (no unauthorized modifications)
**✅ DSMIL Mode 5:** STANDARD level active
**✅ TPM Attestation:** Device 16 operational
**✅ Audit Trail:** All AI queries logged to Device 48
**🔶 Covert Edition:** 25% utilization (can be enhanced)

**From Vault Analysis:**
- Current system: JRTC1 Training Environment compliant
- For SCI/SAP: Requires Level 4 (COMPARTMENTED) upgrade
- Enhancement plan: 4 weeks to full Covert Edition utilization

**Recommendation:** Current configuration adequate for unified AI platform. Covert Edition enhancements optional for production classified workloads.

---

## XV. PERFORMANCE COMPARISON

### Backend Performance Matrix

| Backend | Latency | Throughput | Cost/1K tok | Privacy | DSMIL Attested |
|---------|---------|------------|-------------|---------|----------------|
| **Local DeepSeek** | 10-35s | 32.8 tok/s | $0 | Local | ✅ Yes |
| **Claude Code** | 2-5s | ~100 tok/s | $0.009 | Cloud | ❌ No |
| **Gemini Flash** | 1-3s | ~150 tok/s | $0 | Cloud | ❌ No |
| **OpenAI GPT-4** | 3-8s | ~50 tok/s | $0.020 | Cloud | ❌ No |

**Conclusion:** Use local for privacy/cost, Claude Code for quality, Gemini for speed/multimodal.

---

## XVI. SUMMARY

**You now have:**
- ✅ Local AI with hardware attestation (DeepSeek R1, 32.8 tok/sec)
- ✅ Web interface on port 9876
- ✅ Auto-start enabled (systemd)
- ✅ DSMIL Mode 5 operational (84 devices)
- ✅ Vault integrity verified
- ✅ All changes documented

**Ready to add:**
- ⏳ Claude Code API wrapper
- ⏳ Gemini API wrapper
- ⏳ Unified orchestrator
- ⏳ Intelligent routing

**Your unified platform vision:** One interface, multiple AI backends, intelligent routing, hardware attestation, full local compute power integration.

**Next:** Create the sub-agent wrappers and unified orchestrator to complete the platform.

---

**Classification:** JRTC1 Training Environment
**Distribution:** Authorized Personnel Only

**END OF ARCHITECTURE DOCUMENT**
