# NotebookLM Natural Language Usage Guide

Complete guide for using Google NotebookLM through natural language across all DSMIL AI Engine interfaces.

## Overview

NotebookLM is now fully integrated with **5 different invocation methods**, all supporting natural language:

1. ✅ **Natural Language Interface** (conversational, streaming)
2. ✅ **NotebookLM CLI** (direct commands)
3. ✅ **Unified Orchestrator** (automatic routing)
4. ✅ **Smart Router** (pattern detection)
5. ✅ **MCP Server** (tool protocol)

## 1. Natural Language Interface (Recommended)

**Best for:** Interactive conversations, integrated workflows, streaming progress

The Natural Language Interface provides the most seamless experience with automatic intent recognition, streaming progress, and full integration with the coding assistant.

### Usage

```bash
# Start interactive session
python3 02-ai-engine/natural_language_interface.py

# Or single command
python3 02-ai-engine/natural_language_interface.py "summarize my research documents"
```

### Natural Language Commands

#### Add Sources
```
"add document /path/to/research.pdf"
"upload document /path/to/specs.md"
"add source /path/to/paper.txt"
"load document /path/to/notes.pdf"
"ingest document /path/to/article.md"
```

#### Summarize
```
"summarize my research documents"
"create a summary of all sources"
"give me a summary of my papers"
"summarize everything"
```

#### Create FAQ
```
"create a FAQ"
"create a FAQ from my documents"
"generate FAQ from sources"
"make a FAQ for the team"
```

#### Create Study Guide
```
"create a study guide"
"create study material from my sources"
"make a study guide"
"study guide from all documents"
```

#### Synthesize Sources
```
"synthesize all documents"
"compare sources and find connections"
"cross-reference my research"
"find connections between documents"
"compare all sources"
```

#### Create Executive Briefing
```
"create an executive briefing"
"create briefing from sources"
"executive summary of research"
"create brief for stakeholders"
```

#### List Sources
```
"list my sources"
"show my documents"
"what sources do I have?"
"show all documents"
"list my research"
```

### Features

- ✅ **Streaming Progress**: Real-time progress bars and status updates
- ✅ **Intent Recognition**: Automatically detects NotebookLM vs code tasks
- ✅ **Context Management**: Compressed output saves context tokens
- ✅ **Conversation History**: Maintains conversation across commands
- ✅ **Error Handling**: Graceful degradation with helpful messages

### Example Session

```
You: add document /papers/ml-research.pdf
🤔 Analyzing...
🔍 Processing with NotebookLM (Gemini 2.0 Flash)...
✅ Adding Document complete

You: summarize my research
🤔 Analyzing...
🔍 Processing with NotebookLM (Gemini 2.0 Flash)...
✅ Summarizing Sources complete

[Compressed summary appears here - 500 tokens max]
```

## 2. NotebookLM CLI

**Best for:** Quick commands, scripting, automation

Direct command-line interface with pattern-based NL understanding.

### Usage

```bash
# Add document
python3 02-ai-engine/notebooklm_cli.py "add document /path/to/file.pdf"

# Summarize
python3 02-ai-engine/notebooklm_cli.py "summarize my research papers"

# Create FAQ
python3 02-ai-engine/notebooklm_cli.py "create a FAQ"

# List sources
python3 02-ai-engine/notebooklm_cli.py "list my sources"

# Synthesize
python3 02-ai-engine/notebooklm_cli.py "synthesize all documents"
```

### Pattern Recognition

The CLI understands variations:
- "add document" = "load file" = "upload source" = "ingest paper"
- "summarize" = "create summary" = "give me a summary"
- "create FAQ" = "make FAQ" = "generate FAQ"
- etc.

## 3. Unified Orchestrator

**Best for:** Python integration, programmatic access

The orchestrator automatically routes queries to NotebookLM when appropriate.

### Usage

```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# Automatic routing - detects NotebookLM intent
result = orch.query("summarize my research documents")

# Check routing
print(f"Routed to: {result['backend']}")  # → "notebooklm"
print(f"Task mode: {result['routing']['task_mode']}")  # → "summarize"
print(result['response'])

# Explicit source management
result = orch.query(
    "Add research paper",
    add_source={"file_path": "/path/to/paper.pdf", "title": "ML Research"}
)

# Query with specific sources
result = orch.query(
    "What are the key findings?",
    source_ids=["abc123", "def456"]
)
```

### Automatic Detection

The orchestrator automatically routes these queries to NotebookLM:
- "summarize my documents" → NotebookLM (mode: summarize)
- "create FAQ" → NotebookLM (mode: faq)
- "synthesize sources" → NotebookLM (mode: synthesis)
- "executive briefing" → NotebookLM (mode: briefing)

## 4. Smart Router

**Best for:** Understanding routing logic, custom integrations

The Smart Router provides pattern detection for query routing.

### Usage

```python
from smart_router import SmartRouter

router = SmartRouter()

# Detect NotebookLM queries
decision = router.route("summarize my research papers")

print(decision)
# {
#   "model": "notebooklm",
#   "reason": "notebooklm_task",
#   "task_mode": "summarize",
#   "explanation": "NotebookLM task detected: summarize",
#   "web_search": False
# }
```

### Detection Keywords

**Actions:** summarize, create faq, study guide, synthesize, briefing, analyze documents

**Artifacts:** sources, documents, papers, notes, research materials

**Operations:** add source, create notebook, query sources, compare sources

**Modes:** notebooklm, notebook, document analysis, research assistant

## 5. MCP Server

**Best for:** MCP protocol clients, external integrations

Full MCP protocol support with 13 tools.

### Available Tools

```
notebooklm_add_source           - Add document source
notebooklm_query                - Query with natural language
notebooklm_summarize            - Generate summary
notebooklm_create_faq           - Create FAQ
notebooklm_create_study_guide   - Create study guide
notebooklm_synthesize           - Synthesize sources
notebooklm_create_briefing      - Create briefing
notebooklm_create_notebook      - Create notebook
notebooklm_list_sources         - List sources
notebooklm_list_notebooks       - List notebooks
notebooklm_remove_source        - Remove source
notebooklm_clear_sources        - Clear all sources
notebooklm_status               - Get status
```

### MCP Configuration

Automatically configured in `mcp_servers_config.json`:

```json
{
  "notebooklm": {
    "command": "python3",
    "args": ["03-mcp-servers/notebooklm/notebooklm_mcp_server.py"],
    "env": {
      "PYTHONPATH": "/home/user/LAT5150DRVMIL"
    }
  }
}
```

## Comparison Matrix

| Feature | NL Interface | NotebookLM CLI | Orchestrator | Smart Router | MCP Server |
|---------|--------------|----------------|--------------|--------------|------------|
| **Natural Language** | ✅ Full | ✅ Patterns | ✅ Auto | ✅ Detection | ⚠️ Tools |
| **Streaming** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No |
| **Progress Bars** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No |
| **Context Mgmt** | ✅ Yes | ❌ No | ⚠️ Partial | ❌ No | ❌ No |
| **Conversation** | ✅ Multi-turn | ❌ Single | ❌ Single | ❌ N/A | ❌ Stateless |
| **Routing** | ✅ Auto | ❌ Direct | ✅ Auto | ✅ Detection | ❌ Direct |
| **Python API** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ⚠️ MCP |
| **CLI** | ✅ Yes | ✅ Yes | ⚠️ Partial | ⚠️ Partial | ❌ No |
| **Best For** | Interactive | Quick cmds | Integration | Routing | MCP clients |

## Configuration

**NotebookLM works out-of-the-box** with a hardcoded API key fallback!

**No configuration required** - just start using it immediately.

**Optional:** Set your own API key to override:

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
# or
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**Priority order:**
1. Provided API key (programmatic)
2. GOOGLE_API_KEY environment variable
3. GEMINI_API_KEY environment variable
4. Hardcoded fallback (automatic)

## Tips & Best Practices

### 1. Choose the Right Method

- **Exploring/Learning**: Use Natural Language Interface (interactive, streaming)
- **Quick Tasks**: Use NotebookLM CLI (fastest for single commands)
- **Python Scripts**: Use Unified Orchestrator (automatic routing, clean API)
- **Custom Logic**: Use Smart Router (understand routing decisions)
- **External Tools**: Use MCP Server (protocol compliance)

### 2. Natural Language Tips

**Be conversational:**
```
✅ "summarize my research documents"
✅ "create a FAQ for the team"
✅ "find connections between sources"
```

**Not too technical:**
```
❌ "execute notebooklm_summarize with params source_ids=[]"
✅ "summarize all my sources"
```

### 3. Source Management

**Always add sources first:**
```
1. "add document /papers/research.pdf"
2. "add document /papers/survey.pdf"
3. "summarize my research"  ← Now has sources
```

### 4. Compressed Output

All methods use compressed output to save context:
- Summaries: ~500 tokens
- FAQs: ~700 tokens
- Study guides: ~700 tokens
- Synthesis: ~600 tokens
- Briefings: ~400 tokens

This prevents context window pollution while preserving key information.

### 5. Error Handling

**Out-of-the-box:** NotebookLM now works immediately with hardcoded API key fallback.

If you see an error about missing API key (rare):
```
❌ Error: NotebookLM not available. Please set GOOGLE_API_KEY environment variable.
```

**Solution:** This should only happen if the hardcoded key fails. Try setting your own:
```bash
export GOOGLE_API_KEY="your-key-here"
```

## Advanced Usage

### Combined Workflows

Use Natural Language Interface to combine NotebookLM with coding:

```
You: add document /specs/requirements.pdf
✅ Document added

You: summarize the requirements
✅ [Compressed summary]

You: implement the authentication system from those requirements
📋 Planning task...
⚡ Executing plan...
✅ Task completed!
```

### ACE-FCA Integration

NotebookLM integrates with ACE-FCA workflows:

```python
from notebooklm_subagent import NotebookLMSubagent

# Research phase
agent = NotebookLMSubagent()
result = agent.execute({
    "action": "synthesis"
})

# Returns compressed output (< 600 tokens)
print(result.compressed_output)
```

### Context Optimization

Natural Language Interface automatically optimizes context:

```python
from natural_language_interface import NaturalLanguageInterface

nl = NaturalLanguageInterface()

# Get context stats
stats = nl.get_context_statistics()
print(f"Context usage: {nl.get_context_utilization()}%")

# Force compaction if needed
nl.force_context_compaction(target_pct=50.0)
```

## Troubleshooting

### "NotebookLM not available"
**Cause:** Extremely rare - hardcoded API key failed and no env var set
**Solution:** Set your own API key: `export GOOGLE_API_KEY="your-key"`
**Note:** This should never happen under normal circumstances

### "No sources available"
**Cause:** No documents added yet
**Solution:** Add sources first with "add document /path/to/file"

### Intent not recognized
**Cause:** Ambiguous command
**Solution:** Use clearer keywords (summarize, FAQ, sources, etc.)

### Pattern matching issues
**Check patterns:** See Smart Router detection keywords above

## Examples

### Research Workflow
```bash
# 1. Add sources
python3 natural_language_interface.py "add document /papers/paper1.pdf"
python3 natural_language_interface.py "add document /papers/paper2.pdf"

# 2. Synthesize
python3 natural_language_interface.py "synthesize all sources"

# 3. Create deliverables
python3 natural_language_interface.py "create a FAQ"
python3 natural_language_interface.py "create an executive briefing"
```

### Team Collaboration
```python
from unified_orchestrator import UnifiedAIOrchestrator

orch = UnifiedAIOrchestrator()

# Add team docs
orch.query("add docs", add_source={
    "file_path": "/docs/architecture.md",
    "title": "Architecture Overview"
})

# Generate team resources
faq = orch.query("create FAQ for new developers")
study = orch.query("create study guide for onboarding")
```

## Summary

NotebookLM is now accessible through natural language across the entire DSMIL AI Engine:

1. **5 invocation methods** - Choose the best for your use case
2. **Full NL support** - Conversational, pattern-based understanding
3. **Automatic routing** - Smart Router detects NotebookLM intents
4. **Streaming progress** - Real-time feedback in NL Interface
5. **Context optimization** - Compressed output saves tokens
6. **Seamless integration** - Works with coding, workflows, ACE-FCA

**Try it now:**
```bash
python3 02-ai-engine/natural_language_interface.py
You: summarize my research documents
```

For detailed technical documentation, see `NOTEBOOKLM_INTEGRATION.md`.
