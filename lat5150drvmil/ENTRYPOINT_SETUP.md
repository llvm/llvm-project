# LAT5150 Entrypoint Setup - Complete Guide

## Overview

The LAT5150 entrypoint system has been fully enhanced to handle sudo requirements, path resolution, environment configuration, and dynamic port allocation. This ensures a smooth, production-ready launch experience.

## Quick Start

```bash
# Launch the LAT5150 suite with all pre-flight checks
./lat5150_entrypoint.sh

# Launch with a custom session name
./lat5150_entrypoint.sh my-session

# Enable debug output
LAT5150_DEBUG=1 ./lat5150_entrypoint.sh

# Skip venv setup (for testing)
LAT5150_SKIP_VENV=1 ./lat5150_entrypoint.sh
```

## Key Features

### 1. **No More Port 80 (No More Sudo for API)**
- **Previous behavior**: API required port 80, which needed root/sudo
- **New behavior**: Randomly assigns a port in range 8000-9999
- Port is persistent across sessions (stored in `.lat5150_api_port`)
- Current port shown in tmux panes and startup messages
- Reset port with: `./scripts/get_api_port.sh --reset`

### 2. **Comprehensive Pre-flight Checks**
The entrypoint validates everything before launch:
- ✓ tmux installation
- ✓ Python 3.7+ availability
- ✓ Required script files exist and are executable
- ✓ Virtual environment setup
- ✓ User context preservation (SUDO_USER handling)

### 3. **Absolute Path Resolution**
All paths are resolved to absolute locations, preventing issues when:
- Running from different directories
- Symlinking the script
- Running via sudo with different working directories

### 4. **Environment Variable Configuration**
Automatically exports and configures:
- `LAT5150_ROOT` - Project root directory
- `VIRTUAL_ENV` - Python virtual environment path
- `PYTHONPATH` - Python module search paths (AI Engine + Web Interface)
- `PATH` - Prepends venv/bin for priority
- `LAT5150_SESSION` - tmux session name
- `LAT5150_REAL_USER` - Original user (if run via sudo)
- `LAT5150_REAL_HOME` - Original user's home directory

### 5. **Intelligent Sudo Handling**
- Preserves original user context when run via sudo
- Only requests sudo when actually needed:
  - systemd service management
  - System log viewing
  - DSMIL kernel operations
- API server no longer requires sudo (runs on high port)

### 6. **Auto-fixing Permissions**
- Automatically sets executable bit on critical scripts if missing
- Handles permission issues gracefully with helpful error messages

## File Structure

```
LAT5150DRVMIL/
├── lat5150_entrypoint.sh          # Main entrypoint (enhanced)
├── .lat5150_api_port              # Generated - persistent API port
├── scripts/
│   ├── get_api_port.sh            # NEW - Dynamic port management
│   ├── lat5150_tmux_launcher.sh   # Updated - shows dynamic port
│   └── ensure_venv.sh             # Existing - venv setup
└── lat5150.sh                     # Updated - uses dynamic port
```

## Port Management

### How It Works
1. On first launch, a random port (8000-9999) is selected
2. Port is tested for availability
3. Port is saved to `.lat5150_api_port`
4. Same port is reused across all launches
5. Port is shown in all relevant tmux panes

### Avoiding Common Ports
The system automatically avoids these commonly-used ports:
- 8080 (HTTP alternate)
- 8443 (HTTPS alternate)
- 8888 (Jupyter)
- 9000 (Various services)
- 9090 (Prometheus, etc.)

### Port Commands
```bash
# Get current port
./scripts/get_api_port.sh

# Generate new random port
./scripts/get_api_port.sh --reset

# Check if current port is available
./scripts/get_api_port.sh --check

# Override port via environment variable
LAT5150_API_PORT=9999 ./lat5150.sh start-all
```

## Debug Mode

Enable comprehensive debug output:

```bash
LAT5150_DEBUG=1 ./lat5150_entrypoint.sh
```

Debug mode shows:
- Python version details
- File existence checks
- Environment variable values
- Path resolution steps
- Permission checks
- Venv activation details

## Sudo Requirements

### Operations That Need Sudo
Only these operations will prompt for sudo:
- `systemctl start/stop/enable/disable` (systemd services)
- `journalctl -fu` (viewing system logs)
- `./dsmil.py build-auto` (kernel module compilation)
- `./dsmil.py verify` (checking loaded kernel modules)

### Operations That DON'T Need Sudo Anymore
- Starting the API server (now uses high port)
- Running benchmarks
- Running tests
- Running self-improvement
- Launching dashboard
- Starting tmux session

## Error Handling

The entrypoint provides clear error messages for common issues:

### tmux Not Installed
```
[ERROR] tmux is required but not installed
[INFO] Install with: sudo apt-get install tmux
```

### Python Too Old
```
[ERROR] Python 3.7+ required, found 3.6.9
```

### Missing Virtual Environment
```
[ERROR] Virtual environment not created at /path/to/.venv
[INFO] Run: ./lat5150.sh install
```

### Script Not Executable
```
[WARN] Not executable: /path/to/script.sh
[✓] Fixed permissions: /path/to/script.sh
```

## Integration with Existing Workflows

### Systemd Service
The API service is automatically configured to use the dynamic port:
```ini
[Service]
Environment="LAT5150_API_PORT=8207"
ExecStart=/path/to/venv/bin/python3 unified_tactical_api.py --port 8207
```

### tmux Layout
All tmux panes show the correct port in their help text:
```
Unified Tactical API
====================
Commands:
  ./lat5150.sh api
  curl http://localhost:8207/api/self-awareness

API Port: 8207 (no sudo required)
```

### Status Command
```bash
./lat5150.sh start-all
# Output includes:
#   • API: http://localhost:8207/api/self-awareness
#   • API Port: 8207 (stored in .lat5150_api_port)
```

## Troubleshooting

### Issue: Port Already in Use
```bash
# Reset to new random port
./scripts/get_api_port.sh --reset

# Or manually set a specific port
echo "9876" > .lat5150_api_port
```

### Issue: Permission Denied on Scripts
```bash
# The entrypoint will auto-fix, but you can manually fix:
chmod +x lat5150_entrypoint.sh
chmod +x scripts/*.sh
chmod +x *.py
```

### Issue: Virtual Environment Errors
```bash
# Reinstall venv
rm -rf .venv
./lat5150.sh install
```

### Issue: tmux Session Already Exists
```bash
# Attach to existing session
tmux attach -t lat5150-suite

# Or kill and restart
tmux kill-session -t lat5150-suite
./lat5150_entrypoint.sh
```

## Environment Variables Reference

| Variable | Purpose | Default | Set By |
|----------|---------|---------|--------|
| `LAT5150_DEBUG` | Enable debug output | 0 | User |
| `LAT5150_SKIP_VENV` | Skip venv setup | 0 | User |
| `LAT5150_API_PORT` | Override API port | Random | User or auto |
| `LAT5150_ROOT` | Project root path | Auto-detected | Entrypoint |
| `LAT5150_VENV` | Venv directory | ${LAT5150_ROOT}/.venv | Entrypoint |
| `LAT5150_SESSION` | tmux session name | lat5150-suite | Entrypoint |
| `LAT5150_REAL_USER` | Original user (if sudo) | ${USER} or ${SUDO_USER} | Entrypoint |
| `LAT5150_REAL_HOME` | Original home dir | ${HOME} | Entrypoint |
| `PYTHONPATH` | Python module search | AI_ENGINE:WEB_INTERFACE | Entrypoint |
| `VIRTUAL_ENV` | Active venv path | ${LAT5150_VENV} | Entrypoint |
| `PATH` | Binary search path | venv/bin:${PATH} | Entrypoint |

## Advanced Usage

### Custom Port Range
Edit `scripts/get_api_port.sh`:
```bash
MIN_PORT=10000  # Change to your preferred range
MAX_PORT=19999
```

### Skip Pre-flight Checks (Not Recommended)
```bash
# Only for debugging - skips safety checks
set -euo pipefail
source scripts/ensure_venv.sh
exec scripts/lat5150_tmux_launcher.sh
```

### Launch Multiple Sessions
```bash
# Session 1: Main development
./lat5150_entrypoint.sh dev-session

# Session 2: Testing
LAT5150_API_PORT=9001 ./lat5150_entrypoint.sh test-session

# Session 3: Benchmarking
LAT5150_API_PORT=9002 ./lat5150_entrypoint.sh bench-session
```

## Migration from Previous Setup

If you were using the old entrypoint:

1. ✓ Port 80 references updated to dynamic port
2. ✓ Sudo requirements minimized
3. ✓ All scripts updated to use new port system
4. ✓ Backward compatible with existing workflows

### Breaking Changes
- **Port 80 no longer used**: Update any hardcoded `localhost:80` references
- **Systemd service runs as user**: Previously ran as root
- **API command no longer requires sudo**: Remove `sudo` from scripts

### Migration Checklist
- [ ] Update any external scripts that reference `localhost:80`
- [ ] Reinstall systemd services: `./lat5150.sh install`
- [ ] Update firewall rules if filtering by port
- [ ] Update monitoring tools to check `.lat5150_api_port`

## Testing

Verify the enhanced entrypoint:

```bash
# Test 1: Syntax validation
bash -n lat5150_entrypoint.sh
echo "✓ Syntax OK"

# Test 2: Port generation
./scripts/get_api_port.sh
echo "✓ Port generated"

# Test 3: Port persistence
test "$(./scripts/get_api_port.sh)" = "$(cat .lat5150_api_port)"
echo "✓ Port persists"

# Test 4: Pre-flight checks (dry run)
LAT5150_DEBUG=1 LAT5150_SKIP_VENV=1 ./lat5150_entrypoint.sh 2>&1 | head -20
```

## Support

For issues with the enhanced entrypoint:

1. Enable debug mode: `LAT5150_DEBUG=1 ./lat5150_entrypoint.sh`
2. Check error message details
3. Verify all dependencies installed
4. Check file permissions
5. Review this document

## Summary of Improvements

| Feature | Before | After |
|---------|--------|-------|
| Port allocation | Fixed port 80 | Dynamic 8000-9999 |
| Sudo requirement | Required for API | Only for specific ops |
| Path handling | Relative paths | Absolute paths |
| Error messages | Basic | Comprehensive |
| Pre-flight checks | None | Full validation |
| Environment setup | Manual | Automatic |
| Permission fixes | Manual | Auto-fix |
| User context | Lost with sudo | Preserved |
| Debug capability | None | Full debug mode |
| Port management | N/A | Persistent + resettable |

---

**Last Updated**: 2025-11-18
**Status**: Production Ready
**Tested**: ✓ Syntax, ✓ Port allocation, ✓ Path resolution, ✓ Environment setup
