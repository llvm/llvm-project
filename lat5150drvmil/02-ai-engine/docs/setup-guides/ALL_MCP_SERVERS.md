# DSMIL MCP Server Suite (Security Hardened)

**6 Model Context Protocol Servers** - All with comprehensive security hardening

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version:** 1.0.0
**Last Updated:** 2025-11-06

---

## Overview

Complete suite of MCP servers for Claude Desktop / Cursor integration:

1. **DSMIL AI** - AI engine with RAG and DSMIL device access
2. **Sequential Thinking** - Structured multi-step reasoning
3. **Filesystem** - Sandboxed file operations
4. **Memory** - Persistent knowledge graph
5. **Fetch** - Web content with SSRF protection
6. **Git** - Git operations with injection protection

**All servers include:**
- ✅ Token authentication
- ✅ Rate limiting (60 req/min)
- ✅ Input validation
- ✅ Audit logging
- ✅ Sandboxing/restrictions
- ✅ Error handling

---

## 1. DSMIL AI Server

**File:** `dsmil_mcp_server.py`
**Tools:** 11 tools

### Capabilities

- Query AI with automatic RAG augmentation
- Manage RAG knowledge base (add files/folders, search)
- List AI models (5 models: fast, code, quality, uncensored, large)
- Get PQC status from TPM device
- Get DSMIL device information (84 devices)
- Security status monitoring

### Example Usage

```
Use dsmil_ai_query to ask: "Explain ML-KEM-1024"
Use dsmil_rag_add_folder to index: "/path/to/docs"
Use dsmil_get_status
```

### Security Features

- Query validation (10k char limit, pattern detection)
- Path validation for RAG files
- Sandboxing (only $HOME, /tmp, /var/tmp)
- Audit logging

---

## 2. Sequential Thinking Server

**File:** `sequential_thinking_server.py`
**Tools:** 3 tools
**Based on:** https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking

### Capabilities

- Break down complex problems into steps
- Revise and refine thinking dynamically
- Branch into alternative reasoning paths
- Track thought sessions with history
- Adjust total thoughts estimate on-the-fly

### Example Usage

```
Use sequential_thinking with:
{
  "thought": "First, let's analyze the requirements...",
  "thoughtNumber": 1,
  "totalThoughts": 5,
  "nextThoughtNeeded": true
}
```

### Security Features

- Thought validation (10k char limit)
- Session limit (100 sessions max, prevents memory exhaustion)
- Rate limiting per session
- Audit logging

### Tools

1. **sequential_thinking** - Add thought step
2. **get_thinking_session** - Retrieve session history
3. **list_thinking_sessions** - List all sessions

---

## 3. Filesystem Server

**File:** `filesystem_server.py`
**Tools:** 8 tools
**Based on:** https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

### Capabilities

- Read/write files
- List/create directories
- Delete/move files
- Get file metadata
- Search files with glob patterns

### Example Usage

```
Use read_file: {"path": "/home/user/doc.txt"}
Use write_file: {"path": "/home/user/out.txt", "content": "..."}
Use list_directory: {"path": "/home/user"}
Use search_files: {"pattern": "*.py", "path": "/home/user"}
```

### Security Features

- **Path validation** - Blocks /etc, /root, /boot, /sys, /proc
- **Sandboxing** - Restricted to $HOME, /tmp, /var/tmp
- **File size limits** - 10MB max
- **Extension filtering** - Only allowed file types
- **Symlink protection** - Path resolution before validation
- **Audit logging** - All operations logged

### Tools

1. **read_file** - Read text file
2. **write_file** - Write file
3. **list_directory** - List directory contents
4. **create_directory** - Create directory
5. **delete_file** - Delete file/directory
6. **move_file** - Move/rename
7. **get_file_info** - Get metadata
8. **search_files** - Glob search

---

## 4. Memory Server

**File:** `memory_server.py`
**Tools:** 5 tools
**Based on:** https://github.com/modelcontextprotocol/servers/tree/main/src/memory

### Capabilities

- Persistent knowledge graph across sessions
- Entities with observations
- Relations between entities
- Search by name/type/observations
- Stored in `~/.dsmil/memory_graph.json`

### Example Usage

```
Use create_entity:
{
  "name": "John Doe",
  "type": "person",
  "observations": ["Prefers Python", "Works on DSMIL"]
}

Use create_relation:
{
  "from": "entity_id_1",
  "to": "entity_id_2",
  "type": "works_with"
}

Use search_entities: {"query": "Python"}
```

### Security Features

- Storage file permissions (0600)
- Input validation on all fields
- Rate limiting
- Audit logging

### Tools

1. **create_entity** - Create entity with observations
2. **add_observations** - Add facts to entity
3. **create_relation** - Create relationship
4. **search_entities** - Search by query
5. **read_graph** - Get graph statistics

---

## 5. Fetch Server

**File:** `fetch_server.py`
**Tools:** 1 tool
**Based on:** https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

### Capabilities

- Fetch web content
- Convert HTML to Markdown
- Support for raw HTML
- Configurable content length

### Example Usage

```
Use fetch_url:
{
  "url": "https://example.com",
  "max_length": 5000,
  "raw": false
}
```

### Security Features ⚠️

- **SSRF Protection** - Blocks internal IPs (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8)
- **DNS validation** - Resolves hostname before fetch
- **URL validation** - Only HTTP/HTTPS allowed
- **Size limits** - 1MB max content
- **Timeout** - 10 second request timeout
- **Rate limiting** - Prevents abuse
- **Audit logging** - All fetches logged

### Blocked IP Ranges

```
10.0.0.0/8       (Private)
172.16.0.0/12    (Private)
192.168.0.0/16   (Private)
127.0.0.0/8      (Loopback)
169.254.0.0/16   (Link-local)
::1/128          (IPv6 loopback)
fc00::/7         (IPv6 private)
```

### Tools

1. **fetch_url** - Fetch and convert web content

---

## 6. Git Server

**File:** `git_server.py`
**Tools:** 7 tools
**Based on:** https://github.com/modelcontextprotocol/servers/tree/main/src/git

### Capabilities

- Git status, log, diff, branch
- Stage files (git add)
- Commit changes
- Switch branches

### Example Usage

```
Use git_status: {"repo": "/path/to/repo"}
Use git_log: {"repo": "/path/to/repo", "max_count": 10}
Use git_diff: {"repo": "/path/to/repo"}
Use git_add: {"repo": "/path/to/repo", "files": ["file1.py", "file2.py"]}
Use git_commit: {"repo": "/path/to/repo", "message": "Fix bug"}
```

### Security Features

- **Command injection protection** - No shell execution, subprocess only
- **Path validation** - Sandboxed repo paths
- **Input validation** - Commit messages validated
- **Branch name sanitization** - Alphanumeric + "-_/" only
- **Timeout** - 30 second command timeout
- **Rate limiting**
- **Audit logging**

### Tools

1. **git_status** - Working tree status
2. **git_log** - Commit history
3. **git_diff** - Show changes
4. **git_branch** - List branches
5. **git_add** - Stage files
6. **git_commit** - Commit changes
7. **git_checkout** - Switch branches

---

## Installation

### 1. Install MCP Library

```bash
pip3 install mcp
```

### 2. Install Additional Dependencies

```bash
pip3 install requests beautifulsoup4 html2text
```

### 3. Configure Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dsmil-ai": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/dsmil_mcp_server.py"],
      "env": {"PYTHONPATH": "/home/user/LAT5150DRVMIL"}
    },
    "sequential-thinking": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/sequential_thinking_server.py"]
    },
    "filesystem": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/filesystem_server.py"]
    },
    "memory": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/memory_server.py"]
    },
    "fetch": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/fetch_server.py"]
    },
    "git": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/git_server.py"]
    }
  }
}
```

### 4. Restart Claude Desktop

---

## Security Configuration

All servers share the same security configuration:

**Config File:** `~/.dsmil/mcp_security.json`
**Audit Log:** `~/.dsmil/mcp_audit.log`

### Generate Auth Token (Optional)

```python
from mcp_security import get_security_manager
security = get_security_manager()
token = security.generate_token()
print(f"Token: {token}")
```

### View Security Status

```
Use dsmil_security_status  # From DSMIL AI server
```

### Monitor Audit Log

```bash
tail -f ~/.dsmil/mcp_audit.log
```

---

## Security Summary

| Server | Authentication | Rate Limit | Path Validation | SSRF Protection | Command Injection Protection | Audit Logging |
|--------|---------------|------------|-----------------|-----------------|------------------------------|---------------|
| DSMIL AI | ✅ | ✅ | ✅ | N/A | N/A | ✅ |
| Sequential Thinking | ✅ | ✅ | N/A | N/A | N/A | ✅ |
| Filesystem | ✅ | ✅ | ✅ | N/A | N/A | ✅ |
| Memory | ✅ | ✅ | ✅ | N/A | N/A | ✅ |
| Fetch | ✅ | ✅ | ✅ | ✅ | N/A | ✅ |
| Git | ✅ | ✅ | ✅ | N/A | ✅ | ✅ |

---

## Tool Count Summary

- **DSMIL AI**: 11 tools
- **Sequential Thinking**: 3 tools
- **Filesystem**: 8 tools
- **Memory**: 5 tools
- **Fetch**: 1 tool
- **Git**: 7 tools

**Total: 35 tools** across 6 servers

---

## References

- **MCP Specification:** https://modelcontextprotocol.io/
- **Official MCP Servers:** https://github.com/modelcontextprotocol/servers
- **DSMIL Documentation:** `/home/user/LAT5150DRVMIL/00-documentation/`
- **Security Documentation:** `MCP_SECURITY.md`

---

## Troubleshooting

### Server Won't Start

```bash
# Check dependencies
pip3 install mcp requests beautifulsoup4 html2text

# Test server directly
python3 /path/to/server.py
```

### Rate Limit Issues

Edit `~/.dsmil/mcp_security.json`:

```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 120
  }
}
```

### Check Audit Log

```bash
grep -E "WARNING|ERROR" ~/.dsmil/mcp_audit.log
```

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Author:** DSMIL Integration Framework
**Version:** 1.0.0
