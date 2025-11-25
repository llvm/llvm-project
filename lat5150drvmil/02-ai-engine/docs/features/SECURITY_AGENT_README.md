# Security Agent - Autonomous Security Testing

Autonomous security assessment agent inspired by **Pensar APEX**, implementing AI-driven penetration testing workflows.

## 🎯 Overview

The Security Agent performs comprehensive security assessments using:
- **Autonomous multi-phase testing** (Reconnaissance → Analysis → Reporting)
- **Dynamic tool enumeration** from JSON descriptors
- **AI-powered vulnerability analysis** (when AI model provided)
- **Structured security findings** with severity classification
- **LOCAL-FIRST design** - No external API dependencies

## 🚀 Quick Start

### Basic Usage

```bash
# Run security assessment
python3 security_agent.py example.com

# Or with HTTPS
python3 security_agent.py https://example.com
```

### Test the Tool Registry

```bash
# List available security tools
python3 -c "
from security_agent import ToolRegistry
registry = ToolRegistry()
registry.print_status()
"
```

## 📁 Architecture

```
02-ai-engine/
├── security_agent.py              # Main agent (700+ lines)
│   ├── SecurityAgent              # Autonomous assessment orchestrator
│   ├── ToolRegistry               # Dynamic tool discovery
│   ├── ToolDescriptor             # Tool metadata
│   ├── SecurityFinding            # Individual findings
│   └── SecurityReport             # Structured reports
│
└── security_tools/
    ├── README.md                  # Complete tool documentation
    ├── create_tool_descriptor.py  # Interactive tool generator
    │
    ├── tool_descriptors/          # Tool metadata (JSON)
    │   ├── nmap.json             # Port scanning
    │   ├── nikto.json            # Web vulnerability scanning
    │   ├── gobuster.json         # Directory brute-forcing
    │   ├── testssl.json          # SSL/TLS testing
    │   ├── whatweb.json          # Website fingerprinting
    │   └── nuclei.json           # Fast vulnerability scanner
    │
    └── tool_scripts/              # Custom tool wrappers
        └── (add your scripts here)
```

## 🔍 Assessment Phases

### Phase 1: RECONNAISSANCE
Passive and active information gathering:
- DNS enumeration (`nslookup`, `dig`, `whois`)
- HTTP header analysis (security headers)
- SSL/TLS certificate inspection
- Host availability checking
- Technology fingerprinting

**Tools used:** curl, openssl, nslookup, whois, ping + any custom tools

### Phase 2: ANALYSIS
Security vulnerability identification:
- SSL/TLS configuration analysis
- Security header validation
- DNS misconfiguration detection
- AI-powered vulnerability reasoning (optional)
- Custom tool execution based on phase

**Tools used:** All tools with `"phase": ["analysis"]`

### Phase 3: REPORTING
Structured report generation:
- Severity-based finding classification
- Executive summary with priority actions
- Detailed findings with evidence
- JSON export for automation
- Remediation recommendations

## 🛠️ Tool System

### How Tool Discovery Works

1. **Enumerate** - Agent scans `security_tools/tool_descriptors/` for JSON files
2. **Validate** - Checks if tool command exists in PATH
3. **Select** - Filters tools by assessment phase
4. **Execute** - Runs tools with appropriate profiles
5. **Parse** - Extracts findings using severity keywords

### Tool Descriptor Format

```json
{
  "name": "nmap",
  "description": "Network mapper for port scanning",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "nmap",
  "required": false,
  "args": {
    "basic": "-sV -T4 --top-ports 100 {target}",
    "aggressive": "-A -T4 -p- {target}",
    "stealth": "-sS -T2 {target}"
  },
  "default_profile": "basic",
  "timeout": 300,
  "output_format": "text",
  "parse_output": true,
  "severity_keywords": {
    "critical": ["backdoor", "shell"],
    "high": ["vulnerable", "CVE-"],
    "medium": ["misconfigured", "weak"],
    "low": ["deprecated"],
    "info": ["open", "filtered"]
  }
}
```

### Adding New Tools

#### Method 1: Interactive Generator (Recommended)

```bash
cd 02-ai-engine/security_tools
python3 create_tool_descriptor.py interactive
```

Follow the prompts to create a descriptor.

#### Method 2: Auto from Help

```bash
python3 create_tool_descriptor.py from-help <tool_name>
```

Automatically extracts tool capabilities from `--help` output.

#### Method 3: Manual JSON

Copy an existing descriptor and modify it for your tool.

### Supported Tool Formats

| Format | Example | Location |
|--------|---------|----------|
| System binaries | `nmap`, `nikto` | System PATH |
| Bash scripts | `my_scan.sh` | `tool_scripts/` |
| Python scripts | `custom_check.py` | `tool_scripts/` |
| Any executable | Perl, Ruby, Go | `tool_scripts/` or PATH |

**Key insight:** JSON descriptors are just **metadata**. Tools can be any executable format.

## 📊 Finding Classification

### Severity Levels

| Level | Icon | Description | Example |
|-------|------|-------------|---------|
| **Critical** | 🔴 | Immediate action required | Remote code execution, backdoors |
| **High** | 🟠 | Remediate soon | XSS, SQL injection, weak crypto |
| **Medium** | 🟡 | Plan fixes | Missing headers, misconfigurations |
| **Low** | 🔵 | Minor issues | Version disclosure, deprecated features |
| **Info** | ℹ️ | Informational | Technology stack, open ports |

### Built-in Analysis

The agent includes built-in analyzers for:

1. **SSL/TLS Issues**
   - Weak signature algorithms (SHA-1)
   - Certificate expiration
   - Protocol vulnerabilities

2. **Security Headers**
   - Missing HSTS
   - Missing X-Frame-Options
   - Missing Content-Security-Policy
   - Missing X-Content-Type-Options

3. **DNS Configuration**
   - Multiple A records
   - Subdomain enumeration
   - DNSSEC validation

4. **AI Analysis** (when model provided)
   - Autonomous vulnerability reasoning
   - Context-aware recommendations
   - CVE identification

## 🔗 Integration with AI Engine

### Using with UnifiedAIOrchestrator

```python
from unified_orchestrator import UnifiedAIOrchestrator
from security_agent import SecurityAgent

# Initialize with AI model
orchestrator = UnifiedAIOrchestrator(enable_ace=True)
security_agent = SecurityAgent(ai_model=orchestrator)

# Run assessment with AI analysis
report = security_agent.assess_target("example.com")
security_agent.print_report(report)
```

### AI-Powered Analysis

When an AI model is provided, the agent can:
- Reason about vulnerability chains
- Identify complex attack vectors
- Provide context-aware remediation
- Detect subtle misconfigurations
- Suggest exploitation paths (for authorized testing)

## 📈 Output Formats

### Console Output

```
🔒 Security Assessment Starting: example.com
⚠️  Authorized testing only - ensure you have permission!
📋 Phases: reconnaissance, analysis, reporting

🔍 Phase 1: Reconnaissance
   → DNS lookup...
   → HTTP headers...
   → SSL/TLS certificate...
   ✓ Gathered 6 data points

🔬 Phase 2: Security Analysis
   ✓ Identified 5 findings

📊 Phase 3: Report Generation
   ✓ Report generated

======================================================================
🔒 SECURITY ASSESSMENT REPORT
======================================================================

Target: example.com
Timestamp: 2025-11-06T12:34:56
Phases: reconnaissance, analysis, reporting

Security Assessment Summary for example.com

Total Findings: 5
  - Critical: 0
  - High: 0
  - Medium: 3
  - Low: 1
  - Informational: 1

Priority Actions:
  🟡 MEDIUM: Plan fixes for 3 medium-severity finding(s)
  ✅ No critical or high-severity issues detected

----------------------------------------------------------------------
DETAILED FINDINGS
----------------------------------------------------------------------

🟡 MEDIUM FINDINGS (3):
----------------------------------------------------------------------

1. Missing Strict-Transport-Security Header
   Category: headers
   Description: HSTS header not found - site may be vulnerable to SSL stripping
   Evidence: Strict-Transport-Security header not present
   Recommendation: Add 'Strict-Transport-Security: max-age=31536000' header
...
```

### JSON Export

```json
{
  "target": "example.com",
  "timestamp": "2025-11-06T12:34:56",
  "phases_completed": ["reconnaissance", "analysis", "reporting"],
  "total_findings": 5,
  "findings_by_severity": {
    "critical": 0,
    "high": 0,
    "medium": 3,
    "low": 1,
    "info": 1
  },
  "findings": [
    {
      "title": "Missing Strict-Transport-Security Header",
      "description": "HSTS header not found...",
      "severity": "medium",
      "category": "headers",
      "evidence": "...",
      "recommendation": "...",
      "cve_ids": []
    }
  ]
}
```

## 🔒 Security & Ethics

### ⚠️ AUTHORIZED TESTING ONLY

**CRITICAL:** Only test systems you own or have explicit written authorization to test.

- Unauthorized security testing is **illegal**
- Always obtain written permission before testing
- Document all authorization
- Follow responsible disclosure practices
- Use least intrusive tools first
- Avoid aggressive scans on production systems
- Respect rate limits and avoid DoS

### Responsible Use

This tool is designed for:
- ✅ Authorized penetration testing
- ✅ Bug bounty programs
- ✅ Your own infrastructure
- ✅ Security research (with permission)
- ✅ Educational purposes (authorized targets)

NOT for:
- ❌ Unauthorized testing
- ❌ Malicious activity
- ❌ Vulnerability exploitation without permission
- ❌ Credential theft
- ❌ Data exfiltration

## 🎓 Advanced Usage

### Custom Assessment Workflow

```python
from security_agent import SecurityAgent, SecurityPhase

agent = SecurityAgent()

# Run only reconnaissance
report = agent.assess_target(
    "example.com",
    phases=[SecurityPhase.RECONNAISSANCE]
)

# Run reconnaissance + analysis (no reporting)
report = agent.assess_target(
    "example.com",
    phases=[SecurityPhase.RECONNAISSANCE, SecurityPhase.ANALYSIS]
)
```

### Enable Aggressive Scanning

```python
agent = SecurityAgent(enable_aggressive_scans=True)
report = agent.assess_target("example.com")
```

⚠️ Aggressive scans are more intrusive and may trigger IDS/IPS.

### Custom Tool Integration

```bash
# Create custom tool
cat > security_tools/tool_scripts/my_recon.sh << 'EOF'
#!/bin/bash
TARGET=$1
echo "Custom scan for $TARGET"
# Your custom security checks here
EOF

chmod +x security_tools/tool_scripts/my_recon.sh

# Create descriptor
python3 security_tools/create_tool_descriptor.py interactive
# Follow prompts to describe your tool

# Agent will automatically use it
python3 security_agent.py example.com
```

## 📚 Comparison to APEX

| Feature | APEX | Our Agent | Notes |
|---------|------|-----------|-------|
| Autonomous testing | ✅ | ✅ | Multi-phase workflow |
| AI-powered analysis | ✅ | ✅ | Optional AI model integration |
| Tool integration | ✅ | ✅ | Dynamic tool enumeration |
| Local deployment | ✅ | ✅ | LOCAL-FIRST design |
| Kali container | ✅ | ⚪ | User can add tools manually |
| CLI interface | ✅ | ✅ | Command-line focused |
| Multi-provider AI | ✅ | ✅ | Works with any AI model |
| Structured reporting | ✅ | ✅ | JSON export |

## 🛣️ Roadmap

### Completed ✅
- Multi-phase assessment workflow
- Dynamic tool enumeration
- Tool descriptor system
- Interactive tool generator
- Built-in security analyzers
- Structured reporting (console + JSON)
- Severity-based classification

### Planned 🔮
- [ ] AI-powered tool selection (use ACE-FCA for context)
- [ ] Parallel tool execution (use ParallelAgentExecutor)
- [ ] Exploit chain detection
- [ ] CVE database integration
- [ ] Screenshot capture for web targets
- [ ] Network diagram generation
- [ ] Integration with MCP servers
- [ ] Dashboard UI for reports
- [ ] Continuous monitoring mode
- [ ] Comparison reports (before/after)

## 🤝 Integration Points

### With Existing DSMIL Features

1. **ACE-FCA Context Management**
   - Tool outputs compressed to fit context window
   - Finding summaries optimized for 40-60% utilization

2. **Parallel Execution**
   - Run multiple tools concurrently
   - Use ParallelAgentExecutor for 3-4x speedup

3. **Keyboard Interface**
   - Add security assessment shortcuts
   - Quick target scanning from TUI

4. **Worktree Management**
   - Isolated security test branches
   - Parallel assessments without conflicts

5. **Task Distribution**
   - Assign security tools to optimal agents
   - Load balancing across multiple scans

## 📞 Support

### Getting Help

```bash
# Check tool availability
python3 -c "from security_agent import ToolRegistry; ToolRegistry().print_status()"

# Validate tool descriptor
python3 -m json.tool security_tools/tool_descriptors/nmap.json

# Test tool manually
nmap -sV -T4 example.com
```

### Common Issues

**Issue:** "Tool not found"
```bash
# Check if tool is installed
which nmap

# Install missing tool
sudo apt-get install nmap  # Debian/Ubuntu
brew install nmap          # macOS
```

**Issue:** "Permission denied"
```bash
# Some tools require root/sudo
sudo python3 security_agent.py example.com
```

**Issue:** "Timeout errors"
```bash
# Increase timeout in tool descriptor
"timeout": 600  # 10 minutes
```

## 🎯 Real-World Examples

### Example 1: Quick Web Server Assessment

```bash
python3 security_agent.py https://myapp.example.com
```

Checks:
- SSL/TLS configuration
- Security headers (HSTS, CSP, X-Frame-Options)
- Server version disclosure
- DNS configuration

### Example 2: Comprehensive Network Scan

Add aggressive tool descriptors, then:

```bash
# Install required tools
sudo apt-get install nmap nikto

# Run assessment
python3 security_agent.py example.com
```

### Example 3: Bug Bounty Reconnaissance

```bash
# Create custom recon script
cat > security_tools/tool_scripts/bug_bounty_recon.sh << 'EOF'
#!/bin/bash
TARGET=$1
subfinder -d $TARGET -silent
httpx -l subdomains.txt -silent -status-code
nuclei -l alive.txt -silent
EOF

# Generate descriptor
python3 security_tools/create_tool_descriptor.py interactive

# Run assessment
python3 security_agent.py target.com
```

## 📖 References

- **Pensar APEX**: https://github.com/pensarai/apex
- **GenAI Best Practices**: https://github.com/humanlayer/genai-the-good-parts
- **ACE-FCA Methodology**: Integrated in unified_orchestrator.py
- **OWASP Testing Guide**: https://owasp.org/www-project-web-security-testing-guide/

---

**Remember: With great power comes great responsibility. Use ethically and legally.**

🔒 **AUTHORIZED TESTING ONLY** 🔒
