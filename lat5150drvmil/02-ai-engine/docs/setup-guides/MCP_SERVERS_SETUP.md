# MCP Servers Setup Guide

This guide covers the installation and setup of the five new MCP servers added to the AI interface.

## Prerequisites

- Python 3.10+ (for Metasploit MCP and mcp-for-security)
- Python 3.13+ (for search-tools-mcp)
- Node.js v18+ (for maigret)
- Node.js v22+ (for docs-mcp-server)
- Docker (for maigret and optional mcp-for-security)
- Metasploit Framework (for Metasploit MCP)
- Various security tools (for mcp-for-security: Nmap, Nuclei, SQLmap, etc.)

## Directory Structure

The MCP servers are configured to be installed in:
```
/home/user/LAT5150DRVMIL/03-mcp-servers/
├── search-tools-mcp/
├── MetasploitMCP/
├── mcp-for-security/
└── maigret-reports/  (created automatically)
```

## 1. Search Tools MCP Server

**Purpose**: Advanced code search with symbol analysis and CodeRank algorithm

**Installation**:
```bash
cd /home/user/LAT5150DRVMIL/03-mcp-servers
git clone https://github.com/voxmenthe/search-tools-mcp.git
cd search-tools-mcp
uv sync
```

**Dependencies**:
- `uv` package manager: `pip install uv`
- `kit` CLI tool (for symbol analysis)
- `ripgrep` (for text search): `sudo apt-get install ripgrep` or `brew install ripgrep`

**Configuration**: Already configured in `mcp_servers_config.json`

**Test**:
```bash
uv run mcp dev main.py
```

---

## 2. Docs MCP Server

**Purpose**: Documentation indexing and search with optional vector embeddings

**Installation**:
This server uses `npx` and doesn't require manual installation. It will be downloaded automatically when first run.

**Optional Setup** (for vector search):
- Set `OPENAI_API_KEY` environment variable for enhanced semantic search
- Alternative providers: Google Gemini, AWS Bedrock, Azure OpenAI, Ollama, LM Studio

**Configuration**: Already configured in `mcp_servers_config.json`

**Web Interface**:
Once running, access the management interface at `http://localhost:6280` to queue documentation scraping jobs.

**Test**:
```bash
npx @arabold/docs-mcp-server@latest
```

---

## 3. Metasploit MCP

**Purpose**: Metasploit Framework integration for authorized security testing and exploitation

**Installation**:
```bash
cd /home/user/LAT5150DRVMIL/03-mcp-servers
git clone https://github.com/GH05TCREW/MetasploitMCP.git
cd MetasploitMCP
pip install -r requirements.txt
```

**Metasploit RPC Setup**:
Before using this MCP server, you must start the Metasploit RPC daemon:
```bash
msfrpcd -P your_password -S -a 127.0.0.1 -p 55553
```

**Configuration**:
1. Edit `/home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json`
2. Update the `MSF_PASSWORD` environment variable with your actual password

**Security Note**:
This tool is for authorized security testing only. Ensure you have proper authorization before using any exploitation features.

**Test**:
```bash
python3 MetasploitMCP.py --transport stdio
```

---

## 4. Maigret MCP

**Purpose**: Username OSINT across social networks and URL parsing for OSINT investigations

**Installation**:
This server uses `npx` and doesn't require manual installation. Docker must be installed and running.

**Docker Setup** (if not already installed):
- **macOS/Windows**: Install Docker Desktop
- **Linux**: Install Docker Engine

**Reports Directory**:
The reports directory will be created automatically at:
```
/home/user/LAT5150DRVMIL/03-mcp-servers/maigret-reports/
```

**Configuration**: Already configured in `mcp_servers_config.json`

**Test**:
```bash
npx mcp-maigret@latest
```

---

## 5. MCP for Security

**Purpose**: Comprehensive security testing toolkit with 23 tools for penetration testing and vulnerability assessment

**Installation**:
```bash
cd /home/user/LAT5150DRVMIL/03-mcp-servers
git clone https://github.com/cyproxio/mcp-for-security.git
cd mcp-for-security
bash start.sh
```

**Included Security Tools** (23 total):

**Reconnaissance & Enumeration:**
- Amass, Alterx, Assetfinder, Cero, Certificate Search (crt.sh), shuffledns

**Web Testing:**
- FFUF, Arjun, Katana, httpx, Waybackurls, Gowitness

**Network Scanning:**
- Nmap, Masscan, SSLScan

**Vulnerability Assessment:**
- SQLmap, Nuclei, WPScan, Smuggler, HTTP Headers Security

**Mobile & Cloud:**
- MobSF (mobile app security), Scout Suite (cloud security auditing)

**Alternative Installation** (Docker - Recommended):
```bash
docker pull cyprox/mcp-for-security
# Follow Docker-specific configuration from repository
```

**Dependencies**:
Each tool has its own dependencies. The `start.sh` script handles basic setup, but you may need to install individual tools. Refer to the repository's tool-specific documentation.

**Configuration**: Already configured in `mcp_servers_config.json`

**Security Note**:
All tools are for authorized security testing only. Ensure you have proper authorization before scanning or testing any systems.

**Test**:
```bash
bash /home/user/LAT5150DRVMIL/03-mcp-servers/mcp-for-security/start.sh
```

---

## Verifying Configuration

After installation, verify the configuration:

```bash
cat /home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json
```

All five servers should be listed:
- `search-tools`
- `docs-mcp-server`
- `metasploit`
- `maigret`
- `security-tools`

## Troubleshooting

### Search Tools MCP
- **Error: uv not found**: Install with `pip install uv`
- **Error: kit not found**: Install the cased-kit package
- **Error: ripgrep not found**: Install ripgrep via your package manager

### Docs MCP Server
- **Port 6280 already in use**: Change the PORT environment variable
- **Node version error**: Upgrade to Node.js 22.x or later

### Metasploit MCP
- **Connection error**: Ensure msfrpcd is running with correct credentials
- **Permission denied**: Check that the MSF_PASSWORD matches the RPC daemon password

### Maigret MCP
- **Docker error**: Ensure Docker is installed and running
- **Reports directory error**: Check write permissions for the reports directory

### MCP for Security
- **Tool not found**: Install the specific security tool (Nmap, Nuclei, SQLmap, etc.)
- **Permission denied**: Some tools require root/sudo privileges for certain operations
- **start.sh fails**: Check individual tool dependencies in the repository documentation

## Usage Examples

Once configured, these servers will be available through the AI interface. Examples:

- **Search Tools**: "Search for all function definitions containing 'encrypt'"
- **Docs MCP**: "Index the Python documentation and search for asyncio examples"
- **Metasploit**: "List available exploit modules for Apache servers" (requires authorization)
- **Maigret**: "Search for username 'johndoe' across social networks"
- **Security Tools**: "Run Nmap scan on 192.168.1.0/24" or "Use Nuclei to scan for vulnerabilities" (requires authorization)

## Security Considerations

⚠️ **Important Security Notes**:

1. **Metasploit MCP & Security Tools**: Only use in authorized penetration testing engagements. Unauthorized use is illegal.
2. **Maigret**: Respect privacy laws and terms of service when conducting OSINT investigations.
3. **Environment Variables**: Never commit sensitive credentials (MSF_PASSWORD, API keys) to version control.
4. **Network Access**: Consider firewall rules for servers exposing network ports.
5. **Security Tool Permissions**: Tools like Nmap, Masscan may require elevated privileges. Use responsibly.

## Support

For issues with individual servers, refer to their respective GitHub repositories:
- [search-tools-mcp](https://github.com/voxmenthe/search-tools-mcp)
- [docs-mcp-server](https://github.com/arabold/docs-mcp-server)
- [MetasploitMCP](https://github.com/GH05TCREW/MetasploitMCP)
- [mcp-maigret](https://github.com/BurtTheCoder/mcp-maigret)
- [mcp-for-security](https://github.com/cyproxio/mcp-for-security)
