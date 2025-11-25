# DSMIL Platform - Complete 3-Week Redesign Plan

**Goal:** ChatGPT-style interface with military green aesthetic, smart routing, auto-coding tools built-in

**Status:** Week 1.1 Complete ✅ (Smart Routing)
**Timeline:** 3 weeks total
**Current:** Day 1 complete

---

## User Requirements (Confirmed)

✅ **Theme:** Military green (keep phosphor aesthetic)
✅ **Redesign:** Full (ChatGPT-style 3-panel layout)
✅ **Web:** Search (Google/DuckDuckGo) + Scraping/archiving
✅ **Advanced:** Hide in Settings panel (Flux, GitHub, Collector)
✅ **Auto-coding:** Built directly into UI (file ops, edit ops, Local Claude Code)

---

## WEEK 1: Backend Intelligence (Smart Routing + Web)

### ✅ Week 1.1: Smart Router (COMPLETE - Day 1)

**Built:**
- `smart_router.py` - Intelligent query routing
- Integrated into `unified_orchestrator.py`
- Code detection: write/create/implement + function/class/script
- Complexity analysis: simple/medium/complex
- Web search detection: latest/news/recent queries

**Test Results:**
- "write function factorial" → DeepSeek Coder ✅
- "what is DSMIL" → DeepSeek R1 ✅
- Routing visible: "💻 deepseek-coder | Code task: function"

**Commit:** GitHub pushed ✅

### ⏳ Week 1.2: Web Search Integration (Days 2-3)

**Goal:** Add Google/DuckDuckGo search when needed

**Implementation:**
```python
# File: 02-ai-engine/web_search.py
class WebSearch:
    def search_duckduckgo(self, query):
        """DuckDuckGo API (privacy-first)"""
        # Use duckduckgo-search library
        pass

    def search_google(self, query):
        """Google Custom Search (backup)"""
        # Use Google API
        pass

    def integrate_results(self, query, search_results, ai_response):
        """Combine web results with AI analysis"""
        pass
```

**Routing:**
```python
# When router.web_search_needed == True:
1. Search web (DuckDuckGo)
2. Get top 5 results
3. Summarize with AI
4. Combine: "Based on web search: ... [AI analysis]"
5. Add citations
```

**UI Display:**
```
💬 DeepSeek R1 | 🌐 Web Search (5 sources)

Based on recent news: [AI summarized content]

Sources:
[1] Title - url.com
[2] Title - url.com
```

### ⏳ Week 1.3: Web Scraping & URL Fetching (Days 4-5)

**Goal:** Scrape webpages, auto-add to RAG

**Implementation:**
```python
# File: 04-integrations/web_scraper.py
class WebScraper:
    def scrape_url(self, url):
        """Fetch and parse webpage"""
        # BeautifulSoup for HTML parsing
        # Convert to markdown
        # Extract main content
        pass

    def add_to_rag(self, url, content):
        """Auto-index scraped content"""
        from rag_manager import RAGManager
        manager = RAGManager()
        manager.add_text(content, source=url)
```

**UI Flow:**
```
User: "Scrape https://arxiv.org/abs/2024.12345"
System:
1. Fetch URL
2. Extract content
3. Add to RAG
4. Response: "✓ Added paper to knowledge base (2,345 tokens)"
```

### ⏳ Week 1.4: Backend Testing (Days 6-7)

**Test Scenarios:**
- [ ] Code routing accuracy (100 test queries)
- [ ] Web search triggers correctly
- [ ] All models accessible
- [ ] Performance benchmarks
- [ ] Error handling

**Benchmarks:**
| Query Type | Expected Model | Time | Success |
|------------|----------------|------|---------|
| "write function" | DeepSeek Coder | 5-15s | ✅ |
| "what is X" | DeepSeek R1 | 3-10s | ✅ |
| "latest news" | R1 + Web | 5-8s | ⏳ |
| Complex code | Qwen Coder | 10-30s | ⏳ |

---

## WEEK 2: UI Complete Redesign

### Week 2.1: New 3-Panel Layout (Days 1-3)

**Design Specs:**

```
┌─────────────────────────────────────────────────────────────┐
│ DSMIL AI PLATFORM    🔒 Local | ✓ Attested | 934K docs [⚙]│
├──────────┬──────────────────────────────────────────────────┤
│ SIDEBAR  │  CHAT AREA                                       │
│ (200px)  │  (flexible width)                                │
│          │                                                  │
│ [+ NEW]  │  ┌──────────────────────────────────────┐       │
│          │  │ USER                        8:45 PM  │       │
│ Today    │  │ Write a Python function to check if  │       │
│ • Chat 1 │  │ a number is prime                    │       │
│ • Chat 2 │  └──────────────────────────────────────┘       │
│          │                                                  │
│ This Wk  │  ┌──────────────────────────────────────┐       │
│ • Chat 3 │  │ AI                          8:45 PM  │       │
│ • Chat 4 │  │ 💻 DeepSeek Coder (code detected)    │       │
│          │  │                                       │       │
│ Last Wk  │  │ ```python            [📋Copy][▶Run]  │       │
│ • Chat 5 │  │ def is_prime(n):                     │       │
│          │  │     if n < 2:                        │       │
│ ──────── │  │         return False                 │       │
│ 📚 RAG   │  │     for i in range(2,int(n**0.5)+1):│       │
│ 207 docs │  │         if n % i == 0:               │       │
│          │  │             return False             │       │
│ [+ Add]  │  │     return True                      │       │
│ [Search] │  │ ```                                   │       │
│          │  │                                       │       │
│ ──────── │  │ ✓ Verified | 3.2s | 142 tokens       │       │
│ 🛠 Tools │  └──────────────────────────────────────┘       │
│ [Edit]   │                                                  │
│ [Create] │  ┌──────────────────────────────────────┐       │
│ [Debug]  │  │ ▸ Type message...          [Send]    │       │
│          │  └──────────────────────────────────────┘       │
└──────────┴──────────────────────────────────────────────────┘
```

**Key Elements:**

**Sidebar (Left - 200px):**
- New Chat button (prominent)
- Chat history grouped by date
- RAG section (doc count + add/search)
- **Tools section (AUTO-CODING):**
  - [Edit Code] - Edit existing files
  - [Create File] - Generate new file
  - [Debug] - Fix bugs in code
  - [Refactor] - Improve code structure

**Chat Area (Center - Flexible):**
- Message bubbles (user/AI)
- Timestamp on each message
- Routing tag after AI responses ("💻 DeepSeek Coder")
- Code blocks with syntax highlighting
- Copy button on code
- Run button for Python/JS
- DSMIL verification badge
- Performance stats (time, tokens)

**Input (Bottom):**
- Multi-line textarea
- Send button
- Shift+Enter for newlines
- Character count (optional)

**Settings (Right - Hidden until clicked):**
- Slides out from right
- Sections: General, RAG, Tools, Advanced, About

**Colors (Military Green):**
- Background: #000 (black)
- Text: #0f0 (phosphor green)
- Accents: #ff0 (yellow for warnings/highlights)
- User messages: #0ff (cyan tint)
- AI messages: #0f0 (green)
- Code blocks: #003300 background, #0f0 text

### Week 2.2: One-Click RAG (Day 4)

**Features:**

**1. File Picker Integration:**
```html
<button onclick="openFolderPicker()">📁 Add Folder to RAG</button>

<script>
async function openFolderPicker() {
    // Use HTML5 File API
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;  // Folder selection
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        await indexFiles(files);
    };
    input.click();
}

async function indexFiles(files) {
    // Show progress
    for (let i = 0; i < files.length; i++) {
        showProgress(`Indexing ${i+1}/${files.length}...`);
        await fetch('/rag/add-file?path=' + files[i].path);
    }
    showSuccess(`✓ Added ${files.length} files to RAG`);
}
</script>
```

**2. Drag & Drop:**
```javascript
// Drop zone on RAG panel
ragPanel.addEventListener('drop', async (e) => {
    e.preventDefault();
    const items = e.dataTransfer.items;

    for (let item of items) {
        if (item.kind === 'file') {
            const entry = item.webkitGetAsEntry();
            if (entry.isDirectory) {
                await processDirectory(entry);
            }
        }
    }
});
```

**3. Progress Feedback:**
```
Adding documents to RAG...
[████████░░░░░░] 45/120 files
Indexed: 12,345 tokens

✓ Complete! Added 120 documents (45,678 tokens)
```

### Week 2.3: Settings Panel (Day 5)

**Structure:**
```
⚙️ SETTINGS

┌─ General
│  ├─ Theme: [Military Green ▼] | [Clean Dark] | [Light]
│  ├─ Model: [Auto (recommended) ▼] | [Always use...]
│  ├─ Show routing info: [✓] After each response
│  └─ Web search: [✓] Auto-search when needed
│
├─ RAG Knowledge Base
│  ├─ Documents: 207 indexed
│  ├─ Tokens: 934,743
│  ├─ [+ Add Folder] [🗑 Clear Database]
│  ├─ [📥 Import] [📤 Export]
│  └─ Auto-index downloads: [✓]
│
├─ Auto-Coding Tools ⭐ NEW
│  ├─ File Operations
│  │  ├─ Workspace: [/home/john/LAT5150DRVMIL ▼]
│  │  ├─ Create backups: [✓] (.bak files)
│  │  └─ [Browse Files]
│  ├─ Quick Actions
│  │  ├─ [Edit File] Opens file picker + AI edit
│  │  ├─ [Create File] Generates new file
│  │  ├─ [Debug Code] Analyze + fix bugs
│  │  └─ [Refactor] Improve code structure
│  └─ Code Preferences
│      ├─ Language: [Auto-detect ▼]
│      ├─ Style: [PEP8 / Google / Custom]
│      └─ Comments: [Verbose ▼] | [Minimal] | [None]
│
├─ Advanced Features (Hidden by default)
│  ├─ Flux Network Provider
│  ├─ GitHub Integration
│  ├─ Paper Collector
│  ├─ Hardware Metrics (NPU/GPU/NCS2)
│  └─ DSMIL Attestation Logs
│
└─ About
   ├─ Version: 8.2 (Smart Routing)
   ├─ Local-First AI Platform
   ├─ DSMIL Mode 5 Verified
   ├─ [View Logs] [Documentation]
   └─ [GitHub Repository]
```

### Week 2.4: UI Testing (Days 6-7)

**Test Every UI Element:**
- [ ] New Chat button creates chat
- [ ] Chat history loads previous chats
- [ ] Model selector works (if not auto)
- [ ] Send button sends message
- [ ] Enter key sends message
- [ ] Shift+Enter adds newline
- [ ] Code copy button copies to clipboard
- [ ] Code run button executes (Python/JS)
- [ ] File picker opens and indexes
- [ ] Drag-drop folder indexes
- [ ] RAG search works
- [ ] Settings panel opens/closes
- [ ] Theme switcher changes colors
- [ ] Auto-coding tools (Edit/Create/Debug) work
- [ ] Routing tags display correctly
- [ ] DSMIL badges show
- [ ] Web search integrates (if implemented)

---

## WEEK 3: Polish & Integration

### Week 3.1: Auto-Coding Tools UI Integration (Days 1-2)

**Tools Panel in Sidebar:**

**1. Edit Existing File:**
```
User clicks [Edit Code]
→ Opens file picker
→ User selects file.py
→ File loads in editor panel (right side)
→ User can:
   • Ask AI to edit: "Add error handling to login()"
   • AI generates OLD/NEW strings
   • Shows diff preview
   • User confirms → Edit applied
   • File saved + backup created
```

**2. Create New File:**
```
User clicks [Create File]
→ Modal: "What should this file do?"
→ User types: "FastAPI endpoint for user auth"
→ AI generates complete file
→ Shows preview with syntax highlighting
→ User can edit/approve
→ Save to project → File created
```

**3. Debug Code:**
```
User clicks [Debug]
→ Paste code or select file
→ AI analyzes for:
   • Bugs
   • Security issues
   • Performance problems
   • Best practice violations
→ Shows issues with fixes
→ User can apply fixes
```

**4. Refactor:**
```
User clicks [Refactor]
→ Select file or paste code
→ AI suggests improvements
→ Shows before/after diff
→ Apply changes
```

**Auto-Coding Architecture:**
```
UI Button Click
    ↓
File Picker (if needed)
    ↓
Send to Local Claude Code
    ↓
local_claude_code.py:
  - Read file (file_operations.py)
  - Plan task (AI planning)
  - Generate edits (edit_operations.py)
  - Apply changes
  - Run tests (tool_operations.py)
    ↓
Show Results in UI:
  - Diff preview
  - Test results
  - Confirm/Reject
```

### Week 3.2: Enhanced Code Features (Days 3-4)

**Syntax Highlighting:**
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/monokai.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

<script>
document.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightElement(block);
});
</script>
```

**Copy Button:**
```javascript
function addCopyButton(codeBlock) {
    const button = document.createElement('button');
    button.textContent = '📋 Copy';
    button.onclick = () => {
        navigator.clipboard.writeText(codeBlock.textContent);
        button.textContent = '✓ Copied';
        setTimeout(() => button.textContent = '📋 Copy', 2000);
    };
    codeBlock.parentElement.appendChild(button);
}
```

**Run Button (Python/JS):**
```javascript
function addRunButton(codeBlock, language) {
    if (language !== 'python' && language !== 'javascript') return;

    const button = document.createElement('button');
    button.textContent = '▶ Run';
    button.onclick = async () => {
        const code = codeBlock.textContent;
        const result = await fetch('/exec?cmd=' + encodeURIComponent(
            language === 'python' ? `python3 -c "${code}"` : `node -e "${code}"`
        ));
        const data = await result.json();
        showOutput(data.stdout || data.stderr);
    };
    codeBlock.parentElement.appendChild(button);
}
```

### Week 3.3: RAG Enhanced Search (Day 5)

**Smart RAG Search in Chat:**
```
User: "What techniques does NSA use for supply chain attacks?"

System:
1. Detects question about indexed knowledge
2. Searches RAG: "NSA supply chain"
3. Finds relevant docs (730ARCHIVE)
4. AI reads context + generates answer
5. Shows citations

Response:
"Based on your indexed NSA documents:

[AI-generated summary using RAG context]

📚 Sources:
• 730ARCHIVEPT1.txt (tokens 1234-5678)
• NSA.pdf (page 45-52)

[View Documents] [Ask Follow-up]"
```

**Auto-RAG Integration:**
```python
# Before sending to AI, check if RAG can help:
if query_matches_rag_content(query):
    rag_context = search_and_retrieve(query, top_k=3)
    enhanced_prompt = f"Context from knowledge base:\n{rag_context}\n\nQuestion: {query}"
    response = ai.generate(enhanced_prompt)
```

### Week 3.4: Final Testing & Deployment (Days 6-7)

**Integration Tests:**
- [ ] End-to-end chat flow
- [ ] Auto-coding: Edit→ Preview→Apply works
- [ ] RAG: Add folder → Index → Search → Get context
- [ ] Web: Search triggers → Results integrate
- [ ] Routing: All query types route correctly
- [ ] Performance: <5s for most queries
- [ ] Mobile: Responsive (bonus)

**Bug Fixes:**
- Fix any discovered issues
- Performance optimization
- Error handling improvements

**Documentation:**
- Update README
- User guide
- API documentation

---

## SUCCESS CRITERIA

### User Experience
- ✅ ChatGPT-level simplicity (type and send)
- ✅ Military green aesthetic maintained
- ✅ Auto-routing (invisible, just works)
- ✅ One-click RAG (pick folder, done)
- ✅ Auto-coding tools in UI (edit/create/debug)
- ✅ Web search integrated
- ✅ All features tested

### Technical
- ✅ Routing accuracy: >95%
- ✅ Response time: <5s average
- ✅ RAG indexing: <30s for 100 docs
- ✅ Code quality: 80-90% Claude level
- ✅ DSMIL attestation: 100%

### Unique Value Props
- **Local-First:** Everything private by default
- **No Guardrails:** Perfect for offensive security research
- **DSMIL Attested:** Cryptographic verification (legal protection)
- **Auto-Coding:** Built-in codebase editing (like Claude Code but local)
- **Smart:** Automatically uses right model for task
- **Free:** Zero API costs

---

## IMPLEMENTATION STATUS

### Week 1: Backend Intelligence
- [x] Day 1: Smart Router ✅
- [ ] Days 2-3: Web Search ⏳
- [ ] Days 4-5: Web Scraping ⏳
- [ ] Days 6-7: Testing ⏳

### Week 2: UI Redesign
- [ ] Days 1-3: 3-Panel Layout ⏳
- [ ] Day 4: One-Click RAG ⏳
- [ ] Day 5: Settings Panel ⏳
- [ ] Days 6-7: UI Testing ⏳

### Week 3: Polish
- [ ] Days 1-2: Auto-Coding UI ⏳
- [ ] Days 3-4: Enhanced Code Features ⏳
- [ ] Day 5: RAG Integration ⏳
- [ ] Days 6-7: Final Testing ⏳

**Current:** Day 1 complete (Smart Routing)
**Next:** Days 2-3 (Web Search)
**Timeline:** On track for 3-week delivery

---

## DIFFERENTIATORS vs ChatGPT/Claude

| Feature | ChatGPT | Claude Code | DSMIL (After Redesign) |
|---------|---------|-------------|----------------------|
| **Privacy** | Cloud | Cloud | 100% Local ✅ |
| **Restrictions** | High | Medium | None ✅ |
| **Cost** | $20/mo | $20/mo | $0 ✅ |
| **Verification** | None | None | DSMIL TPM ✅ |
| **Code Editing** | No | Yes | Yes (Local) ✅ |
| **Web Search** | Yes | Yes (new) | Yes (integrated) ✅ |
| **RAG/Docs** | No | No | 934K tokens ✅ |
| **Offensive Security** | Blocked | Blocked | Allowed ✅ |
| **Speed** | 2-5s | 3-8s | 3-15s ⚠️ |
| **Quality** | 95% | 100% | 80-90% ⚠️ |

**Unique Selling Points:**
1. Offensive security research (exploit dev, malware analysis) - no blocks
2. DSMIL attestation (legal audit trail)
3. LOCAL-FIRST (complete privacy)
4. Zero cost (unlimited usage)
5. Auto-coding tools (built-in codebase editing)

---

## FILES TO CREATE (Week 2-3)

**Week 2:**
- `03-web-interface/clean_ui_v3.html` - New simplified UI
- `03-web-interface/static/styles_green.css` - Military green theme
- `03-web-interface/static/app.js` - Modern UI logic
- `03-web-interface/static/highlight-green.css` - Code syntax (green theme)

**Week 3:**
- `02-ai-engine/web_search.py` - DuckDuckGo/Google integration
- `04-integrations/web_scraper.py` - Enhanced scraping
- `03-web-interface/components/file_picker.html` - Folder selection
- `03-web-interface/components/code_tools.html` - Auto-coding UI
- `03-web-interface/components/settings_panel.html` - Settings sidebar

---

**This 3-week plan delivers a LOCAL-FIRST ChatGPT competitor with:**
- Better privacy
- No restrictions
- DSMIL verification
- Auto-coding tools
- Offensive security support

**Target:** Production-ready by Week 3 Day 7
