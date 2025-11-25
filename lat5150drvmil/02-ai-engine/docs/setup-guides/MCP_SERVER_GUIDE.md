# DSMIL AI MCP Server Guide

## Overview

The DSMIL AI MCP (Model Context Protocol) Server exposes the DSMIL AI Engine and its capabilities to MCP-compatible clients like Claude Desktop, Cursor, and other AI development tools.

This enables external AI assistants to:
- Query the DSMIL AI engine with automatic RAG context
- Manage the RAG knowledge base
- Access DSMIL hardware security device information
- Check Post-Quantum Cryptography status
- Interact with 84 standard DSMIL devices

## Features

### 10 MCP Tools Exposed

1. **dsmil_ai_query** - Query AI with RAG augmentation
2. **dsmil_rag_add_file** - Add file to knowledge base
3. **dsmil_rag_add_folder** - Add folder to knowledge base
4. **dsmil_rag_search** - Search knowledge base
5. **dsmil_get_status** - Get system status
6. **dsmil_list_models** - List available AI models
7. **dsmil_rag_list_documents** - List indexed documents
8. **dsmil_rag_stats** - Get RAG statistics
9. **dsmil_pqc_status** - Get Post-Quantum Crypto status
10. **dsmil_device_info** - Get DSMIL device information

## Installation

### 1. Install MCP Library

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
pip3 install -r mcp_requirements.txt
```

### 2. Configure MCP Client

#### For Claude Desktop

Edit your Claude Desktop configuration file:

**Linux/macOS:**
```bash
~/.config/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

Add the DSMIL AI server configuration:

```json
{
  "mcpServers": {
    "dsmil-ai": {
      "command": "python3",
      "args": [
        "/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/home/user/LAT5150DRVMIL"
      }
    }
  }
}
```

**Note:** Replace `/home/user/LAT5150DRVMIL` with your actual installation path.

#### For Cursor

Add to Cursor's MCP settings:

```json
{
  "mcp": {
    "servers": {
      "dsmil-ai": {
        "command": "python3",
        "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_mcp_server.py"]
      }
    }
  }
}
```

## Usage Examples

Once configured, you can use DSMIL AI tools from your MCP client:

### Query the AI

```
Use dsmil_ai_query to ask: "What are the key features of ML-KEM-1024?"
```

The AI will automatically search the RAG knowledge base and augment its response with relevant context.

### Add Documentation to Knowledge Base

```
Use dsmil_rag_add_folder to index: "/home/user/LAT5150DRVMIL/00-documentation"
```

This recursively indexes all supported files (PDF, TXT, MD, LOG, C, H, PY, SH).

### Search Knowledge Base

```
Use dsmil_rag_search to find: "Post-Quantum Cryptography"
```

Returns relevant documents and snippets.

### Check System Status

```
Use dsmil_get_status
```

Returns:
- Ollama connection status
- Available models (5 models: fast, code, quality, uncensored, large)
- RAG statistics (document count, token count)
- DSMIL device status
- Mode 5 security level
- Guardrails configuration

### Get PQC Status

```
Use dsmil_pqc_status
```

Returns TPM Post-Quantum Cryptography capabilities:
- ML-KEM-1024 (FIPS 203) - Key Encapsulation
- ML-DSA-87 (FIPS 204) - Digital Signatures
- AES-256-GCM (FIPS 197) - Symmetric Encryption
- SHA-512 (FIPS 180-4) - Hashing
- MIL-SPEC compliance status

### Get Device Information

```
Use dsmil_device_info with device_id: "0x8000"
```

Returns information about the TPM Control device.

Omit device_id to list all 84 standard devices.

## Architecture

```
┌─────────────────────────────────────┐
│   MCP Client (Claude/Cursor/etc)    │
│                                     │
│  User: "Query DSMIL AI about PQC"  │
└─────────────┬───────────────────────┘
              │ MCP Protocol
              │ (stdio)
┌─────────────▼───────────────────────┐
│     DSMIL MCP Server                │
│  (dsmil_mcp_server.py)              │
│                                     │
│  - 10 MCP Tools                     │
│  - Request routing                  │
│  - Error handling                   │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│   DSMIL AI Engine                   │
│  (dsmil_ai_engine.py)               │
│                                     │
│  - Ollama integration               │
│  - 5 models (Phi-3, DeepSeek, etc)  │
│  - RAG system (automatic)           │
│  - Model routing                    │
└─────────────┬───────────────────────┘
              │
       ┌──────┴──────┐
       │             │
┌──────▼──────┐ ┌───▼──────────────┐
│ Ollama      │ │ RAG System       │
│ (Local LLM) │ │ (Token-based)    │
│             │ │                  │
│ 5 Models    │ │ ~/.rag_index     │
└─────────────┘ └──────────────────┘
```

## Available Models

The DSMIL AI engine supports 5 models:

1. **fast** - Phi-3-mini-128k-instruct (3.8B params)
   - Quick responses, low resource usage
   - Good for general queries

2. **code** - DeepSeek-Coder-V2-Lite-Instruct (16B params)
   - Specialized for code generation
   - Excellent for programming tasks

3. **quality** - Llama-3.1-8B-Instruct (8B params)
   - High-quality responses
   - Balanced performance/quality

4. **uncensored** - WizardLM-1.0-Uncensored-CodeLlama-34b (34B params)
   - No content filtering
   - Technical/security research

5. **large** - Qwen2.5-32B-Instruct (32B params)
   - Highest capability model
   - Complex reasoning and analysis

## RAG (Retrieval Augmented Generation)

The RAG system automatically augments AI queries with relevant context from the knowledge base.

**Supported File Types:**
- PDF, TXT, MD, LOG
- C, H (C/C++ source)
- PY (Python)
- SH (Shell scripts)

**Index Location:**
- `~/.rag_index/` (automatically created)

**Search Method:**
- Token-based (not vector embeddings)
- Fast and efficient for documentation

## Security

### Classification
- UNCLASSIFIED // FOR OFFICIAL USE ONLY
- No classified information should be processed

### Mode 5 Compliance
- Operates within DSMIL Mode 5 security constraints
- Guardrails enforce content policies
- All operations logged

### PQC Compliance
- TPM device uses MIL-SPEC Post-Quantum algorithms:
  - ML-KEM-1024 (Security Level 5)
  - ML-DSA-87 (Security Level 5)
  - AES-256-GCM
  - SHA-512

## Troubleshooting

### MCP Server Won't Start

Check dependencies:
```bash
pip3 install mcp
python3 -c "import mcp; print(mcp.__version__)"
```

### Ollama Not Connected

Start Ollama:
```bash
systemctl start ollama
# or
ollama serve
```

Check status:
```bash
curl http://localhost:11434/api/tags
```

### RAG Not Working

Initialize RAG system:
```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 test_rag.py
```

Check index:
```bash
ls -la ~/.rag_index/
```

### Permission Issues

Ensure executable permissions:
```bash
chmod +x /home/user/LAT5150DRVMIL/02-ai-engine/dsmil_mcp_server.py
```

## Testing

### Test MCP Server Directly

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 dsmil_mcp_server.py
```

The server communicates via stdio, so you won't see output unless sending MCP protocol messages.

### Test AI Engine

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 ai_query.py "Test query"
```

### Test RAG System

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 test_rag.py
```

## Integration with Context7

Context7 (https://context7.com/) provides up-to-date documentation for LLMs. To integrate:

1. **Use Context7 MCP Server** to fetch documentation
2. **Use DSMIL RAG** to index the fetched documentation
3. **Query DSMIL AI** with automatic Context7 documentation context

Example workflow:
```
1. Context7 fetches latest Python 3.12 docs
2. DSMIL RAG indexes the documentation
3. Ask DSMIL AI: "How do I use Python 3.12 match statements?"
4. Response includes Context7 documentation + AI explanation
```

## Performance

- **Cold start:** ~2-3 seconds (engine initialization)
- **Query latency:** 100-500ms (fast model) to 2-5s (large model)
- **RAG search:** <100ms for most queries
- **Concurrent requests:** Supported (async/await)

## Limitations

- Requires local Ollama installation
- Models must be pulled: `ollama pull <model>`
- RAG is token-based (not vector embeddings)
- Maximum context: 128k tokens (varies by model)

## Version History

- **v1.0.0** - Initial MCP server release
  - 10 MCP tools
  - RAG integration
  - PQC status
  - Device information

## Support

For issues or questions:
- Check logs: `~/.rag_index/` and Ollama logs
- Test components individually (AI, RAG, devices)
- Verify MCP configuration in client

## References

- Model Context Protocol: https://modelcontextprotocol.io/
- Context7: https://context7.com/
- DSMIL Documentation: `/home/user/LAT5150DRVMIL/00-documentation/`
- Ollama: https://ollama.ai/

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Author:** DSMIL Integration Framework
**Version:** 1.0.0
