# LAT5150 DRVMIL - Auto-Install & Auto-Start Guide

## Overview

Complete automation for installing and configuring the LAT5150 Unified Tactical API with Atomic Red Team integration. This system:

✓ Auto-installs all dependencies
✓ Auto-starts on every boot
✓ Auto-restarts on failure
✓ Provides Natural Language interface on port 80
✓ Integrates Atomic Red Team (MITRE ATT&CK)
✓ Registers 20 system capabilities

## Quick Start

### One-Command Installation

```bash
cd /home/user/LAT5150DRVMIL/deployment
sudo ./install-unified-api-autostart.sh install
```

This will:
1. Check and install all dependencies (Python, Flask, uvx, etc.)
2. Configure Atomic Red Team MCP server
3. Install SystemD service
4. Enable auto-start on boot
5. Start the service immediately

### Verify Installation

```bash
sudo ./install-unified-api-autostart.sh status
```

### Test the API

```bash
# Check API is responding
curl http://localhost:80/api/self-awareness

# Query Atomic Red Team
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me atomic tests for T1059.002"}'
```

## What Gets Installed

### 1. System Service

**File**: `/etc/systemd/system/lat5150-unified-api.service`

**Features**:
- Runs on port 80 (requires root)
- Auto-starts on boot
- Auto-restarts on failure
- Journal logging
- Security hardening

### 2. Dependencies

**Python Packages**:
- Flask 2.x+
- Flask-CORS
- Other requirements from unified_tactical_api.py

**System Tools**:
- uv/uvx (Rust-based Python package manager)
- atomic-red-team-mcp (via uvx)

### 3. Directory Structure

```
/home/user/LAT5150DRVMIL/
├── 02-ai-engine/
│   └── atomic_red_team_api.py        # Python API wrapper
├── 03-web-interface/
│   ├── unified_tactical_api.py       # Main NL interface
│   ├── capability_registry.py        # 20 capabilities registered
│   ├── natural_language_processor.py # NL query processing
│   └── ATOMIC_RED_TEAM_NL_USAGE.md  # Usage documentation
└── 03-mcp-servers/
    └── atomic-red-team-data/         # Test data directory
```

## Service Management

### Start Service

```bash
sudo systemctl start lat5150-unified-api
```

### Stop Service

```bash
sudo systemctl stop lat5150-unified-api
```

### Restart Service

```bash
sudo systemctl restart lat5150-unified-api
```

### Check Status

```bash
sudo systemctl status lat5150-unified-api
```

### View Logs

```bash
# Follow logs in real-time
sudo journalctl -u lat5150-unified-api -f

# View last 50 lines
sudo journalctl -u lat5150-unified-api -n 50
```

### Disable Auto-Start

```bash
sudo systemctl disable lat5150-unified-api
```

### Enable Auto-Start

```bash
sudo systemctl enable lat5150-unified-api
```

## Uninstallation

```bash
sudo ./install-unified-api-autostart.sh remove
```

This will:
1. Stop the service
2. Disable auto-start
3. Remove SystemD service file
4. Reload SystemD daemon

## Natural Language Interface

### Access Points

**Port 80** (default): `http://localhost:80`

### Available Capabilities

The system has **20 registered capabilities**:

#### Atomic Red Team (4 capabilities):
1. `atomic_query_tests` - Query MITRE ATT&CK test cases
2. `atomic_list_techniques` - List all techniques
3. `atomic_refresh` - Update tests from GitHub
4. `atomic_validate` - Validate YAML structure

#### System Capabilities (16 capabilities):
- Code Understanding (Serena LSP)
- Code Manipulation
- Hardware Access (DSMIL)
- Agent Execution (AgentSystems)
- Security & Audit
- System Control & Monitoring

### Example Queries

#### Atomic Red Team Queries:

```bash
# Query by technique ID
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me atomic tests for T1059.002"}'

# Platform-specific search
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find mshta atomics for Windows"}'

# List all techniques
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "List all MITRE ATT&CK techniques"}'

# Refresh tests
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Refresh atomic red team tests"}'
```

#### System Queries:

```bash
# Check system health
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Check system health"}'

# Find code symbol
curl -X POST http://localhost:80/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find the NSADeviceReconnaissance class"}'
```

### Self-Awareness Report

```bash
curl http://localhost:80/api/self-awareness | jq '.'
```

Returns:
```json
{
  "version": "3.0.0",
  "deployment": "LOCAL-FIRST",
  "components": {
    "serena_lsp": true/false,
    "agentsystems": true/false,
    "model_manager": true/false,
    "self_awareness_engine": true,
    "atomic_red_team": true
  },
  "legacy_capabilities": {
    "total_capabilities": 20,
    "by_category": {...}
  }
}
```

## Boot Sequence

### What Happens on Boot

1. **System Boot** → SystemD starts
2. **Network Online** → Wait for network
3. **Service Start** → lat5150-unified-api.service starts
4. **Dependencies Check** → Python, Flask, uvx verified
5. **Components Init** → All 20 capabilities loaded
6. **Atomic Red Team** → MCP server configured
7. **API Ready** → Listening on port 80

### Startup Time

- Typical startup: 3-5 seconds
- Full initialization: 5-10 seconds

### Monitoring Startup

```bash
# Watch startup in real-time
sudo journalctl -u lat5150-unified-api -f
```

Expected output:
```
Starting LAT5150 DRVMIL Unified Tactical API with Atomic Red Team...
📦 LOCAL MODELS: whiterabbit, llama3.2:latest, codellama:latest
✅ Serena LSP initialized
✅ AgentSystems runtime initialized
✅ Local model provider initialized (default)
✅ Atomic Red Team initialized (MITRE ATT&CK techniques)
Initialized 20 capabilities
✅ Ready for natural language commands
Running on http://0.0.0.0:80
```

## Configuration

### Service File Location

`/etc/systemd/system/lat5150-unified-api.service`

### Environment Variables

Set in service file:
- `PYTHONUNBUFFERED=1` - Real-time logging
- `PYTHONPATH` - Include source directories
- `TACTICAL_PORT=80` - API port
- `ART_MCP_TRANSPORT=stdio` - Atomic Red Team transport
- `ART_DATA_DIR` - Test data directory
- `ART_EXECUTION_ENABLED=false` - Safety (execution disabled)

### Modify Configuration

1. Edit service file:
   ```bash
   sudo nano /etc/systemd/system/lat5150-unified-api.service
   ```

2. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart lat5150-unified-api
   ```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u lat5150-unified-api -n 50

# Common issues:
# - Port 80 already in use: sudo lsof -i :80
# - Missing dependencies: Check error in logs
# - Permission issues: Service runs as root
```

### API Not Responding

```bash
# Check if service is running
sudo systemctl status lat5150-unified-api

# Check if port 80 is listening
sudo netstat -tulpn | grep :80

# Test connection
curl -v http://localhost:80/api/self-awareness
```

### Atomic Red Team Not Working

```bash
# Check if uvx is installed
which uvx

# Check data directory exists
ls -la /home/user/LAT5150DRVMIL/03-mcp-servers/atomic-red-team-data

# Test MCP server directly
uvx atomic-red-team-mcp --help

# Check API component
curl http://localhost:80/api/self-awareness | jq '.components.atomic_red_team'
# Should return: true
```

### Reinstall Service

```bash
# Remove existing
sudo ./install-unified-api-autostart.sh remove

# Reinstall
sudo ./install-unified-api-autostart.sh install
```

## Advanced Usage

### Running on Different Port

Edit service file and change:
```ini
Environment="TACTICAL_PORT=5001"
```

And update ExecStart:
```ini
ExecStart=/usr/bin/python3 ... --port 5001
```

### Enable Test Execution (Caution!)

Edit service file and change:
```ini
Environment="ART_EXECUTION_ENABLED=true"
```

**WARNING**: Only enable in isolated test environments!

### Custom Models

Edit service file and update:
```ini
ExecStart=/usr/bin/python3 ... --local-models your-model,another-model
```

## Security Considerations

### Service Security Features

- `PrivateTmp=true` - Isolated tmp directory
- `ProtectSystem=strict` - Read-only system files
- `ProtectHome=read-only` - Read-only home (except workspace)
- `NoNewPrivileges=false` - Allow capability changes (for uvx)

### Network Security

- Service binds to `0.0.0.0:80` (all interfaces)
- Consider firewall rules for production:
  ```bash
  sudo ufw allow 80/tcp
  ```

### Test Execution Safety

- Atomic Red Team execution **DISABLED by default**
- Requires explicit environment variable change
- Only enable in controlled test VMs
- Monitor all executed tests

## Support

### Documentation

- Main docs: `/home/user/LAT5150DRVMIL/00-documentation/`
- API docs: `/home/user/LAT5150DRVMIL/03-web-interface/ATOMIC_RED_TEAM_NL_USAGE.md`
- Service file: `/etc/systemd/system/lat5150-unified-api.service`

### Logs

- SystemD journal: `journalctl -u lat5150-unified-api`
- Service status: `systemctl status lat5150-unified-api`

### Verification Checklist

After installation, verify:
- [ ] Service is active: `systemctl is-active lat5150-unified-api`
- [ ] Service is enabled: `systemctl is-enabled lat5150-unified-api`
- [ ] API responds: `curl http://localhost:80/api/self-awareness`
- [ ] Atomic Red Team loaded: Check self-awareness report
- [ ] Can query tests: Test example query
- [ ] Logs are clean: `journalctl -u lat5150-unified-api -n 20`

## Summary

### Before Installation

- Fresh system or existing LAT5150 install
- Root access (sudo)
- Internet connection (for dependencies)

### After Installation

- ✅ Unified Tactical API running on port 80
- ✅ Atomic Red Team fully integrated
- ✅ 20 capabilities registered
- ✅ Auto-starts on every boot
- ✅ Auto-restarts on failure
- ✅ Natural Language interface ready
- ✅ Journal logging enabled
- ✅ Security hardening applied

### Next Steps

1. Test natural language queries
2. Explore MITRE ATT&CK techniques
3. Integrate with existing workflows
4. Monitor logs and performance
5. Configure firewall if needed

---

**Version**: 1.0.0
**Last Updated**: 2025-11-17
**Maintained By**: LAT5150 DRVMIL Development Team
