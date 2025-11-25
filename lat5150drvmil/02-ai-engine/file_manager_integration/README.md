# DSMIL AI - File Manager Integration

## Overview

This directory contains the integration components that enable **right-click "Open DSMIL AI"** functionality in file managers, allowing you to open the DSMIL AI coding assistant in any local folder.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Right-Click Integration                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  File Manager Context Menu                                  │
│          ↓                                                  │
│  open_dsmil_ai.sh                                           │
│          ↓                                                  │
│  DSMIL Terminal API Server (Unix socket)                    │
│          ↓                                                  │
│  Code Specialist + Code Assistant + Self-Improvement        │
│          ↓                                                  │
│  Interactive Terminal Session                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. **Terminal API Server** (`dsmil_terminal_api.py`)

Unix domain socket server providing IPC for external processes.

**Features**:
- JSON-RPC 2.0 protocol
- Session management per directory
- User-specific socket (`/tmp/dsmil-ai-{uid}.sock`)
- Authentication tokens
- Conversation history
- Project context analysis

**Methods**:
- `create_session`: Create new coding session
- `close_session`: Close session
- `list_sessions`: List active sessions
- `query`: Execute AI query
- `generate_code`: Generate code
- `review_code`: Review code
- `execute_code`: Execute code safely
- `analyze_project`: Analyze project structure
- `get_session_history`: Get conversation history

### 2. **API Client** (`dsmil_api_client.py`)

Python client library for connecting to the API server.

**Usage**:
```python
from dsmil_api_client import DSMILClient

client = DSMILClient()
session = client.create_session("/path/to/project")
response = client.query(session['session_id'], "Generate a Python function")
print(response['response'])
```

### 3. **Context Menu Script** (`open_dsmil_ai.sh`)

Universal script that opens DSMIL AI in any directory.

**Features**:
- Auto-detects selected directory from file manager
- Checks/starts API server automatically
- Opens terminal with interactive session
- Project analysis and context loading
- Conversation history management

### 4. **Installation Script** (`install_context_menu.sh`)

Installs context menu integration for all detected file managers.

**Supported File Managers**:
- **Nautilus** (GNOME Files)
- **Thunar** (XFCE)
- **Dolphin** (KDE)
- **Nemo** (Cinnamon)
- **Caja** (MATE)

## Installation

### Quick Install

```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine/file_manager_integration
./install_context_menu.sh
```

This will:
1. Detect all installed file managers
2. Install "Open DSMIL AI" context menu for each
3. Create necessary directories
4. Set up symbolic links

### Manual Install

#### Nautilus (GNOME)
```bash
mkdir -p ~/.local/share/nautilus/scripts
ln -s /path/to/open_dsmil_ai.sh ~/.local/share/nautilus/scripts/"Open DSMIL AI"
```

#### Thunar (XFCE)
```bash
mkdir -p ~/.local/share/Thunar/sendto
# Create desktop file (see install script)
```

#### Dolphin (KDE)
```bash
mkdir -p ~/.local/share/kservices5/ServiceMenus
# Create service menu (see install script)
```

### Uninstall

```bash
./install_context_menu.sh --uninstall
```

## Usage

### From File Manager

1. Open your file manager (Nautilus, Thunar, etc.)
2. Navigate to any project directory
3. **Right-click** on the folder
4. Select **"Open DSMIL AI"** from the context menu
5. Terminal opens with DSMIL AI assistant

### From Command Line

#### Start API Server
```bash
cd /path/to/LAT5150DRVMIL/02-ai-engine
python3 dsmil_terminal_api.py --daemon
```

#### Use API Client
```bash
# Ping server
python3 dsmil_api_client.py ping

# Create session
python3 dsmil_api_client.py create /path/to/project

# Query
python3 dsmil_api_client.py query <session_id> "Generate a Python function"

# Analyze project
python3 dsmil_api_client.py analyze <session_id>

# List sessions
python3 dsmil_api_client.py list
```

#### Direct Script
```bash
./open_dsmil_ai.sh /path/to/project
```

## Interactive Commands

When the terminal session opens, you can use these commands:

| Command | Description |
|---------|-------------|
| `help` or `?` | Show available commands |
| `analyze` | Re-analyze project structure |
| `history` | Show conversation history |
| `generate <lang>` | Generate code in specified language |
| `review <file>` | Review code file |
| `exit` or `quit` | Exit assistant |

### Example Queries

```
DSMIL> Generate a Python function that reads a JSON file and returns a dict

DSMIL> How do I implement error handling in async code?

DSMIL> Explain this code pattern: context managers

DSMIL> Refactor my code to use type hints

DSMIL> Review security vulnerabilities in authentication.py
```

## Project Context

The assistant automatically analyzes your project:

**Detected Information**:
- Project type (Python, Rust, C/C++, JavaScript)
- Git repository status
- File structure
- Dependencies
- Entry points

**Auto-detection**:
- `setup.py` / `pyproject.toml` → Python project
- `Cargo.toml` → Rust project
- `package.json` → JavaScript project
- `CMakeLists.txt` / `Makefile` → C/C++ project

## Session Management

### Sessions

Each directory gets its own session with:
- Unique session ID
- Authentication token
- Conversation history
- Project context
- Working directory

### Persistence

Sessions persist until:
- Explicitly closed (`close_session`)
- Server restart
- Timeout (configurable)

### Multiple Sessions

You can have multiple sessions open simultaneously:
```bash
# Session 1: Project A
DSMIL> (working on project-a/)

# Session 2: Project B
DSMIL> (working on project-b/)
```

## Security

### Socket Permissions

Unix domain socket has user-only permissions (0600):
```bash
ls -l /tmp/dsmil-ai-*.sock
srw------- 1 user user 0 Nov 20 12:00 /tmp/dsmil-ai-1000.sock
```

### Authentication

Each session has an authentication token:
- Generated on session creation
- Required for API calls
- Tied to session ID

### Sandboxing

Code execution (if enabled) is sandboxed:
- Limited file system access
- No network access by default
- Resource limits
- Timeout enforcement

### Input Validation

All inputs are validated using `SecurityHardening` from the DSMIL system:
- SQL injection prevention
- Command injection prevention
- Path traversal prevention
- XSS prevention

## Integration with Existing Components

### Code Specialist

Uses specialized coding models:
- **DeepSeek Coder 6.7B**: Fast code generation
- **Qwen 2.5 Coder 7B**: High-quality implementations
- **CodeLlama 70B**: Code review and analysis

### Code Assistant

RAG-enhanced code assistance with:
- Multi-turn conversations
- Context from documentation
- File operations
- Syntax highlighting
- Security scanning
- Performance analysis

### Autonomous Self-Improvement

Monitors and learns from interactions:
- Performance metrics
- Bottleneck detection
- Improvement proposals
- Meta-learning

### 98-Agent System

Access to 98 specialized agents:
- Strategic planning
- Development
- Infrastructure
- Security
- QA
- Documentation
- Operations

## Troubleshooting

### Server Not Starting

**Problem**: `Cannot connect to DSMIL API server`

**Solution**:
```bash
# Check if server is running
ps aux | grep dsmil_terminal_api

# Check socket
ls -l /tmp/dsmil-ai-*.sock

# Start server manually
python3 dsmil_terminal_api.py

# Check logs
tail -f /tmp/dsmil-terminal-api.log
```

### Context Menu Not Appearing

**Problem**: "Open DSMIL AI" doesn't appear in context menu

**Solution**:
```bash
# Reinstall
./install_context_menu.sh

# Restart file manager
killall nautilus  # or thunar, dolphin, etc.

# Check installation
ls -l ~/.local/share/nautilus/scripts/
```

### Terminal Not Opening

**Problem**: Nothing happens when clicking menu item

**Solution**:
```bash
# Check script permissions
chmod +x open_dsmil_ai.sh

# Test script directly
./open_dsmil_ai.sh /path/to/project

# Check terminal emulator
which gnome-terminal xfce4-terminal konsole xterm
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'code_specialist'`

**Solution**:
```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/LAT5150DRVMIL/02-ai-engine:$PYTHONPATH

# Install dependencies
pip3 install -r requirements.txt

# Check imports
python3 -c "from code_specialist import CodeSpecialist"
```

## Performance

### Startup Time
- **Cold start**: ~2-3 seconds (first time)
- **Warm start**: ~0.5 seconds (server already running)

### Response Time
- **Simple queries**: ~1-2 seconds
- **Code generation**: ~3-5 seconds
- **Complex analysis**: ~5-10 seconds

### Resource Usage
- **Memory**: ~500MB (server + models)
- **CPU**: Minimal when idle
- **Disk**: ~10MB for session data

## Advanced Usage

### Custom Socket Path

```bash
python3 dsmil_terminal_api.py --socket /tmp/my-custom.sock
```

### Programmatic Access

```python
import json
import socket

# Connect to API
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/dsmil-ai-1000.sock')

# Send request
request = {
    "jsonrpc": "2.0",
    "method": "create_session",
    "params": {"working_dir": "/home/user/project"},
    "id": 1
}
sock.sendall(json.dumps(request).encode())

# Receive response
response = json.loads(sock.recv(4096).decode())
print(response['result'])
```

### Integration with IDEs

The API can be integrated with IDEs like VS Code:
- Extension can connect to Unix socket
- Provide in-editor AI assistance
- Context-aware code completion
- Real-time suggestions

## Future Enhancements

### Planned Features

- [ ] Visual Studio Code extension
- [ ] JetBrains IDE plugin
- [ ] Web-based UI
- [ ] Multi-user support
- [ ] Remote API access (SSH tunneling)
- [ ] Voice input support
- [ ] Real-time file watching
- [ ] Automatic refactoring suggestions
- [ ] CI/CD integration
- [ ] Git commit message generation

### Experimental Features

- [ ] Neural code completion
- [ ] Predictive bug detection
- [ ] Auto-generated unit tests
- [ ] Performance profiling
- [ ] Dependency vulnerability scanning

## Contributing

To add support for a new file manager:

1. Identify the context menu location
2. Update `install_context_menu.sh`
3. Add detection logic
4. Create appropriate menu files
5. Test thoroughly

## License

This is part of the LAT5150DRVMIL platform.

## Support

For issues or questions:
- Check the troubleshooting section
- Review the main project documentation
- Open an issue in the repository

---

**Author**: LAT5150DRVMIL AI Platform
**Version**: 1.0.0
**Date**: 2025-11-20
