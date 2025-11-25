# MCP Servers Directory

This directory contains all external MCP (Model Context Protocol) servers for the LAT5150DRVMIL AI interface.

## Quick Start

Run the automated setup script to install all MCP servers:

```bash
cd /home/user/LAT5150DRVMIL/03-mcp-servers
bash setup_mcp_servers.sh
```

This will automatically:
- ✅ Clone all required repositories
- ✅ Install dependencies
- ✅ Set up directory structure
- ✅ Verify prerequisites
- ✅ Provide next steps and test commands

## Installed Servers

After running the setup script, you'll have:

### 1. search-tools-mcp
**Path**: `./search-tools-mcp/`
**Purpose**: Advanced code search with symbol analysis and CodeRank
**Command**: `uv run --directory /path/to/search-tools-mcp main.py`

### 2. docs-mcp-server
**Path**: N/A (npx auto-download)
**Purpose**: Documentation indexing and vector search
**Command**: `npx @arabold/docs-mcp-server@latest`

### 3. MetasploitMCP
**Path**: `./MetasploitMCP/`
**Purpose**: Metasploit Framework integration for security testing
**Command**: `python3 /path/to/MetasploitMCP.py --transport stdio`
**Requires**: Metasploit Framework + msfrpcd running

### 4. maigret
**Path**: N/A (npx auto-download)
**Reports**: `./maigret-reports/`
**Purpose**: Username OSINT across social networks
**Command**: `npx mcp-maigret@latest`
**Requires**: Docker

### 5. mcp-for-security
**Path**: `./mcp-for-security/`
**Purpose**: 23 security testing tools (Nmap, Nuclei, SQLmap, etc.)
**Command**: `bash /path/to/mcp-for-security/start.sh`
**Requires**: Individual tool installations

## Manual Installation

If you prefer to install servers manually, see the detailed guide:
```
/home/user/LAT5150DRVMIL/02-ai-engine/MCP_SERVERS_SETUP.md
```

## Configuration

All servers are configured in:
```
/home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json
```

## Testing Individual Servers

After installation, test each server:

```bash
# Test search-tools-mcp
cd search-tools-mcp && uv run mcp dev main.py

# Test docs-mcp-server
npx @arabold/docs-mcp-server@latest

# Test maigret (requires Docker)
npx mcp-maigret@latest

# Test MetasploitMCP (requires msfrpcd running)
cd MetasploitMCP && python3 MetasploitMCP.py --transport stdio

# Test mcp-for-security
cd mcp-for-security && bash start.sh
```

## Prerequisites

- **Python 3.10+** (for MetasploitMCP and mcp-for-security)
- **Python 3.13+** (for search-tools-mcp)
- **Node.js 18+** (for maigret)
- **Node.js 22+** (for docs-mcp-server)
- **Docker** (for maigret, optional for mcp-for-security)
- **uv** package manager: `pip install uv`
- **ripgrep**: `sudo apt-get install ripgrep` or `brew install ripgrep`

## Security Notes

⚠️ **Important**: The following servers are for authorized security testing ONLY:
- **MetasploitMCP**: Requires authorization for any exploitation or scanning
- **mcp-for-security**: All 23 tools require authorization for target scanning
- **maigret**: Respect privacy laws and terms of service

Unauthorized use of security tools is illegal and unethical.

## Troubleshooting

### "uv not found"
```bash
pip3 install uv
```

### "ripgrep not found"
```bash
# Ubuntu/Debian
sudo apt-get install ripgrep

# macOS
brew install ripgrep
```

### "Docker not found"
Install Docker Desktop (macOS/Windows) or Docker Engine (Linux):
https://docs.docker.com/get-docker/

### "msfrpcd connection failed"
1. Ensure Metasploit Framework is installed
2. Start the RPC daemon:
   ```bash
   msfrpcd -P your_password -S -a 127.0.0.1 -p 55553
   ```
3. Update `MSF_PASSWORD` in `mcp_servers_config.json`

### Security tool not found
For mcp-for-security, individual tools must be installed separately. See:
https://github.com/cyproxio/mcp-for-security

## Support

For issues with specific servers, refer to their GitHub repositories:
- [search-tools-mcp](https://github.com/voxmenthe/search-tools-mcp)
- [docs-mcp-server](https://github.com/arabold/docs-mcp-server)
- [MetasploitMCP](https://github.com/GH05TCREW/MetasploitMCP)
- [mcp-maigret](https://github.com/BurtTheCoder/mcp-maigret)
- [mcp-for-security](https://github.com/cyproxio/mcp-for-security)

## Directory Structure

```
03-mcp-servers/
├── README.md                    # This file
├── setup_mcp_servers.sh         # Automated setup script
├── search-tools-mcp/            # Code search server
├── MetasploitMCP/               # Metasploit integration
├── mcp-for-security/            # Security tools suite
└── maigret-reports/             # OSINT reports directory
```

## Updates

To update servers to the latest version:

```bash
# Update git-based servers
cd search-tools-mcp && git pull && uv sync
cd ../MetasploitMCP && git pull && pip3 install -r requirements.txt
cd ../mcp-for-security && git pull

# npx-based servers auto-update to @latest
```
