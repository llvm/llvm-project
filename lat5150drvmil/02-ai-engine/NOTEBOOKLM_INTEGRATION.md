# Google NotebookLM Integration for DSMIL AI Engine

## Overview

Complete integration of Google NotebookLM-style functionality into the DSMIL AI Engine, providing document-based research, synthesis, and analysis capabilities with natural language invocation.

## Architecture

### Components

1. **NotebookLM Wrapper** (`sub_agents/notebooklm_wrapper.py`)
   - Core NotebookLM agent using Gemini 2.0 Flash API
   - Document ingestion (PDF, Markdown, Text)
   - Multi-source Q&A with grounding
   - Summary, FAQ, study guide generation
   - Source synthesis and executive briefings

2. **NotebookLM MCP Server** (`03-mcp-servers/notebooklm/notebooklm_mcp_server.py`)
   - MCP protocol server for NotebookLM tools
   - 13 MCP tools for document operations
   - Full integration with MCP ecosystem

3. **NotebookLM Subagent** (`notebooklm_subagent.py`)
   - ACE-FCA compliant subagent
   - Context-isolated execution
   - Compressed output (prevents context pollution)
   - Workflow integration (Research → Plan → Implement → Verify)

4. **Smart Router Integration** (`smart_router.py`)
   - Automatic detection of NotebookLM queries
   - Natural language pattern matching
   - Task mode detection (summarize, FAQ, study guide, etc.)

5. **Unified Orchestrator Integration** (`unified_orchestrator.py`)
   - Seamless backend routing to NotebookLM
   - Source management operations
   - Query mode execution

6. **Natural Language CLI** (`notebooklm_cli.py`)
   - Conversational interface
   - Pattern-based command parsing
   - User-friendly natural language commands

## Features

### Document Operations
- ✅ Add sources (text, PDF, Markdown)
- ✅ Create notebooks with source collections
- ✅ List and manage sources
- ✅ Remove sources or clear all

### Research & Analysis
- ✅ Multi-source Q&A with grounding
- ✅ Document summarization
- ✅ FAQ generation
- ✅ Study guide creation
- ✅ Source synthesis (find connections across documents)
- ✅ Executive briefing generation

### Integration
- ✅ Automatic routing via Smart Router
- ✅ Natural language invocation
- ✅ MCP protocol support
- ✅ ACE-FCA workflow integration
- ✅ 2M token context support (Gemini 2.0 Flash)

## Usage

### 1. Configuration

**NotebookLM now works out-of-the-box!** A Google API key is hardcoded as a fallback.

**No configuration required** - just start using immediately.

**Optional:** Set your own Google API key to override:
```bash
export GOOGLE_API_KEY="your-google-api-key-here"
# or
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**Priority order:**
1. Provided `api_key` parameter
2. `GOOGLE_API_KEY` environment variable
3. `GEMINI_API_KEY` environment variable
4. **Hardcoded fallback** (automatic)

### 2. Natural Language CLI

```bash
# Add sources
python3 notebooklm_cli.py "add document /path/to/research.pdf"
python3 notebooklm_cli.py "load file /path/to/notes.txt"

# Ask questions
python3 notebooklm_cli.py "what are the main findings?"
python3 notebooklm_cli.py "how does the authentication system work?"

# Generate content
python3 notebooklm_cli.py "create a FAQ"
python3 notebooklm_cli.py "generate a study guide"
python3 notebooklm_cli.py "summarize all sources"
python3 notebooklm_cli.py "synthesize my documents"
python3 notebooklm_cli.py "create an executive briefing"

# Manage sources
python3 notebooklm_cli.py "list my sources"
python3 notebooklm_cli.py "status"
```

### 3. Unified Orchestrator

```python
from unified_orchestrator import UnifiedAIOrchestrator

orchestrator = UnifiedAIOrchestrator()

# Automatic routing to NotebookLM
result = orchestrator.query("summarize my research documents")

# Explicit source addition
result = orchestrator.query(
    "Research paper analysis",
    add_source={"file_path": "/path/to/paper.pdf", "title": "Research Paper"}
)

# Query with specific sources
result = orchestrator.query(
    "What are the key findings?",
    source_ids=["abc123", "def456"]
)
```

### 4. ACE-FCA Workflow

```python
from notebooklm_subagent import NotebookLMSubagent

subagent = NotebookLMSubagent()

# Research phase: Ingest documents
result = subagent.execute({
    "action": "add_source",
    "file_path": "/path/to/spec.pdf",
    "title": "System Specification"
})

# Analysis phase: Synthesize findings
result = subagent.execute({
    "action": "synthesis"
})
# Returns compressed output (< 600 tokens)
print(result.compressed_output)

# Planning phase: Create briefing
result = subagent.execute({
    "action": "briefing"
})
```

### 5. MCP Server

The NotebookLM MCP server is automatically configured in `mcp_servers_config.json`:

```json
{
  "notebooklm": {
    "command": "python3",
    "args": ["/home/user/LAT5150DRVMIL/03-mcp-servers/notebooklm/notebooklm_mcp_server.py"],
    "env": {
      "PYTHONPATH": "/home/user/LAT5150DRVMIL"
    }
  }
}
```

Available MCP tools:
- `notebooklm_add_source` - Add document source
- `notebooklm_query` - Query sources with natural language
- `notebooklm_summarize` - Generate summary
- `notebooklm_create_faq` - Create FAQ
- `notebooklm_create_study_guide` - Create study guide
- `notebooklm_synthesize` - Synthesize across sources
- `notebooklm_create_briefing` - Create executive briefing
- `notebooklm_create_notebook` - Create notebook
- `notebooklm_list_sources` - List all sources
- `notebooklm_list_notebooks` - List all notebooks
- `notebooklm_remove_source` - Remove a source
- `notebooklm_clear_sources` - Clear all sources
- `notebooklm_status` - Get status and stats

### 6. Smart Router

The Smart Router automatically detects NotebookLM queries:

```python
from smart_router import SmartRouter

router = SmartRouter()

# Automatic detection
decision = router.route("summarize my research papers")
# Returns: {"model": "notebooklm", "task_mode": "summarize", ...}

decision = router.route("create a FAQ from my documents")
# Returns: {"model": "notebooklm", "task_mode": "faq", ...}

decision = router.route("what are the key findings in the reports?")
# Returns: {"model": "notebooklm", "task_mode": "qa", ...}
```

Detection keywords:
- **Actions**: summarize, create faq, study guide, synthesize, briefing, analyze documents
- **Artifacts**: sources, documents, papers, notes, research materials
- **Operations**: add source, create notebook, query sources, compare sources
- **Modes**: notebooklm, notebook, document analysis, research assistant

## Storage

Sources and notebooks are stored in:
```
~/.dsmil/notebooklm/
  ├── sources.json      # Source metadata and content
  └── notebooks.json    # Notebook configurations
```

## Technical Details

### Model
- **Backend**: Google Gemini 2.0 Flash Experimental
- **Context Window**: 2M tokens (2,000,000 tokens)
- **API**: Google Generative AI SDK
- **Cost**: Free (Student tier)

### Document Processing
- **Text**: Direct ingestion
- **PDF**: PyPDFLoader (langchain)
- **Markdown**: Native support
- **Storage**: Local JSON files

### ACE-FCA Compliance
- **Context Isolation**: Dedicated context window per subagent
- **Output Compression**: 400-700 tokens depending on mode
- **Phase Integration**: Research, Analysis, Planning, Verification
- **Human Review**: Checkpoints at compaction boundaries

## Integration Points

### Smart Router Detection
File: `02-ai-engine/smart_router.py:133-167`

Detects NotebookLM tasks and returns routing decision with task mode.

### Unified Orchestrator Routing
File: `02-ai-engine/unified_orchestrator.py:227-260`

Routes to NotebookLM backend and handles source management operations.

### MCP Server Registration
File: `02-ai-engine/mcp_servers_config.json:100-107`

Registers NotebookLM as MCP server in the ecosystem.

### ACE Subagent System
File: `02-ai-engine/notebooklm_subagent.py`

Provides ACE-FCA compliant subagent for workflow integration.

## Examples

### Research Workflow

```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# 1. Ingest research papers
orch.query("add research", add_source={
    "file_path": "/papers/ml-paper-1.pdf",
    "title": "Deep Learning Research"
})
orch.query("add research", add_source={
    "file_path": "/papers/ml-paper-2.pdf",
    "title": "Transformer Architecture"
})

# 2. Synthesize findings
result = orch.query("synthesize findings across all papers")
print(result['response'])

# 3. Create FAQ for team
result = orch.query("create a FAQ for the research team")
print(result['response'])

# 4. Executive briefing
result = orch.query("create an executive briefing")
print(result['response'])
```

### Code Documentation Analysis

```python
from notebooklm_subagent import NotebookLMSubagent

agent = NotebookLMSubagent()

# Add documentation sources
agent.execute({
    "action": "add_source",
    "file_path": "/docs/api-reference.md",
    "title": "API Reference"
})
agent.execute({
    "action": "add_source",
    "file_path": "/docs/architecture.md",
    "title": "Architecture Guide"
})

# Query across docs
result = agent.execute({
    "action": "query",
    "prompt": "How does the authentication flow work?",
    "mode": "qa"
})
print(result.compressed_output)

# Create study guide for new developers
result = agent.execute({
    "action": "study_guide"
})
print(result.compressed_output)
```

## Troubleshooting

### API Key Not Set (Extremely Rare)
**Note:** This should never happen - hardcoded API key fallback is automatic.

```
Error: NotebookLM not available. Please set GOOGLE_API_KEY environment variable.
```
**Solution** (if this rare error occurs): Set your own API key:
```bash
export GOOGLE_API_KEY="your-key-here"
```

### Module Not Found
```
ImportError: No module named 'google.generativeai'
```
**Solution**: Install dependencies:
```bash
pip install google-generativeai langchain langchain-community pypdf
```

### File Not Found
```
Error: File not found: /path/to/file.pdf
```
**Solution**: Ensure the file path is correct and accessible.

## Performance

- **Document Ingestion**: ~1-2 seconds per document
- **Query Response**: ~3-5 seconds (depends on context size)
- **Summary Generation**: ~5-10 seconds
- **FAQ Creation**: ~8-12 seconds
- **Study Guide**: ~10-15 seconds
- **Synthesis**: ~8-12 seconds

Times vary based on document size and complexity.

## Limitations

1. **API Availability**: Requires valid Google API key
2. **Cloud Dependency**: NotebookLM uses cloud-based Gemini API (not local)
3. **Document Size**: Large documents (>1M tokens) may be slow
4. **Rate Limits**: Subject to Google API rate limits

## Future Enhancements

- [ ] Local embeddings for faster retrieval
- [ ] Vector database integration (ChromaDB/Qdrant)
- [ ] Notebook sharing and collaboration
- [ ] Audio/video source support (via Gemini multimodal)
- [ ] Real-time collaboration features
- [ ] Export to various formats (PDF, Markdown, HTML)

## Credits

- **Implementation**: LAT5150DRVMIL AI Platform
- **Powered by**: Google Gemini 2.0 Flash
- **ACE-FCA**: HumanLayer methodology
- **MCP Protocol**: Anthropic

## License

Part of the LAT5150DRVMIL project.
