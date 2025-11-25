# Atomic Red Team - Natural Language Interface Usage

## Overview

Atomic Red Team security testing framework is now fully integrated into the LAT5150 DRVMIL Unified Tactical API with natural language query support.

**Access via**: `http://localhost:5001` (default) or `http://localhost:80` (with --port 80)

## Quick Start

### 1. Start the Unified Tactical API

```bash
cd /home/user/LAT5150DRVMIL/03-web-interface
python3 unified_tactical_api.py
```

Or with custom port:
```bash
python3 unified_tactical_api.py --port 80
```

### 2. Query Using Natural Language

#### Via HTTP API:

```bash
# Query atomic tests
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me atomic tests for T1059.002"}'

# Find Windows tests
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find mshta atomics for Windows"}'

# List all techniques
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List all MITRE ATT&CK techniques"}'
```

#### Via Python:

```python
import requests

# Query atomic tests
response = requests.post(
    'http://localhost:5001/api/query',
    json={'query': 'Show me atomic tests for T1059.002'}
)

result = response.json()
print(f"Success: {result['success']}")
print(f"Tests found: {result['result']['count']}")
```

## Natural Language Triggers

The system automatically detects Atomic Red Team queries using these keywords:

### Primary Keywords:
- `atomic`, `atomic test`, `atomic red team`
- `mitre`, `mitre attack`, `att&ck`
- `security test`, `red team test`
- `adversary technique`

### Technique-Specific:
- Any MITRE ATT&CK technique ID (e.g., `T1059`, `T1003`, `T1105`)

### Example Queries:

```
✓ "Show me atomic tests for T1059.002"
✓ "Find mshta atomics for Windows"
✓ "Search for PowerShell red team tests"
✓ "List all MITRE ATT&CK techniques for macOS"
✓ "What atomic tests are available for Linux?"
✓ "Refresh atomic red team tests"
✓ "Validate this atomic test YAML"
```

## Available Capabilities

### 1. Query Atomic Tests (`atomic_query_tests`)

**Description**: Search for MITRE ATT&CK security test cases

**Parameters**:
- `query` (str): Natural language query or search terms
- `technique_id` (Optional[str]): MITRE ATT&CK technique ID (e.g., T1059.002)
- `platform` (Optional[str]): windows, linux, macos

**Examples**:
```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me atomic tests for T1059.002"
  }'
```

**Response**:
```json
{
  "success": true,
  "capability": "atomic_query_tests",
  "confidence": 0.95,
  "result": {
    "success": true,
    "tests": [...],
    "count": 5,
    "query": "Show me atomic tests for T1059.002",
    "timestamp": "2025-11-17T10:00:00"
  }
}
```

### 2. List Techniques (`atomic_list_techniques`)

**Description**: List all available MITRE ATT&CK techniques with test counts

**Parameters**: None

**Examples**:
```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List all MITRE ATT&CK techniques"}'
```

### 3. Refresh Tests (`atomic_refresh`)

**Description**: Update Atomic Red Team tests from official Red Canary GitHub repository

**Parameters**: None

**Examples**:
```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Refresh atomic red team tests"}'
```

### 4. Validate YAML (`atomic_validate`)

**Description**: Validate atomic test YAML structure against official schema

**Parameters**:
- `yaml_content` (str): YAML content to validate

**Examples**:
```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Validate this atomic test YAML",
    "yaml_content": "attack_technique: T1059.002\n..."
  }'
```

## Integration with Other Systems

### With AI System Integrator (Port 5050):

The dashboard on port 5050 (`ai_gui_dashboard.py`) also includes Atomic Red Team:

```bash
cd /home/user/LAT5150DRVMIL/02-ai-engine
python3 ai_gui_dashboard.py
```

Access via: `http://localhost:5050/api/atomic-red-team/*`

### With MCP Servers:

Atomic Red Team is configured as MCP server #14 in `mcp_servers_config.json`:

```json
{
  "atomic-red-team": {
    "command": "uvx",
    "args": ["atomic-red-team-mcp"],
    "env": {
      "ART_EXECUTION_ENABLED": "false"
    }
  }
}
```

## Architecture

```
User Natural Language Query
    ↓
Unified Tactical API (port 5001 or 80)
    ↓
NaturalLanguageProcessor (keyword detection)
    ↓
CapabilityRegistry (atomic capabilities)
    ↓
UnifiedTacticalAPI._execute_capability()
    ↓
AtomicRedTeamAPI (Python wrapper)
    ↓
MCP Server (uvx atomic-red-team-mcp)
    ↓
Red Canary Atomic Red Team Repository
    ↓
Formatted Response (JSON)
```

## Self-Awareness Report

Check if Atomic Red Team is loaded:

```bash
curl http://localhost:5001/api/self-awareness | jq '.components.atomic_red_team'
```

Expected output: `true`

## Security Notes

- **Test Execution DISABLED by default** (`ART_EXECUTION_ENABLED=false`)
- Requires explicit configuration change to enable
- Only enable in controlled, isolated test environments
- All operations are logged and auditable

## Troubleshooting

### "Atomic Red Team not available"

Check initialization:
```bash
# View logs when starting unified_tactical_api.py
python3 unified_tactical_api.py

# Expected output:
# ✅ Atomic Red Team initialized (MITRE ATT&CK techniques)
```

If not initialized:
1. Verify `atomic_red_team_api.py` is in `02-ai-engine/`
2. Check `uvx` is installed: `which uvx`
3. Verify MCP server config in `mcp_servers_config.json`

### No Results Found

Try:
- Using MITRE ATT&CK technique IDs (e.g., `T1059.002`)
- Platform filters: `Windows`, `Linux`, `macOS`
- Refreshing tests: `"Refresh atomic red team tests"`

## Complete Example Session

```bash
# 1. Start API
python3 unified_tactical_api.py --port 80

# 2. Query for Windows PowerShell tests
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find PowerShell atomic tests for Windows"}'

# 3. Get specific technique
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me atomic tests for T1059.001"}'

# 4. List all techniques
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List all MITRE ATT&CK techniques"}'

# 5. Refresh repository
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Refresh atomic tests"}'
```

## Next Steps

- Explore MITRE ATT&CK framework: https://attack.mitre.org/
- Red Canary Atomic Red Team: https://github.com/redcanaryco/atomic-red-team
- Review test YAML structure for custom tests
- Enable execution in isolated test VMs (requires configuration change)

---

**Last Updated**: 2025-11-17
**API Version**: 3.0.0
**Atomic Red Team MCP Server**: Latest (uvx)
