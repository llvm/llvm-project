# Security Tools for Autonomous Security Agent

This directory contains tool descriptors and scripts for the Security Agent to enumerate and use during autonomous security assessments.

## Directory Structure

```
security_tools/
├── README.md                    # This file
├── tool_descriptors/            # JSON descriptors for security tools
│   ├── nmap.json               # Example: Nmap descriptor
│   ├── nikto.json              # Example: Nikto descriptor
│   └── ...                     # Add more tool descriptors
└── tool_scripts/               # Custom tool wrappers/scripts
    ├── custom_recon.sh         # Example custom script
    └── ...                     # Add more scripts
```

## Tool Descriptor Format

Tool descriptors are JSON files that describe how to use a security tool:

```json
{
  "name": "nmap",
  "description": "Network mapper for port scanning and service detection",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "nmap",
  "required": false,
  "args": {
    "basic": "-sV -T4 {target}",
    "aggressive": "-A -T4 {target}",
    "stealth": "-sS -T2 {target}",
    "udp": "-sU {target}"
  },
  "default_profile": "basic",
  "timeout": 300,
  "output_format": "text",
  "parse_output": true,
  "severity_keywords": {
    "critical": ["exploit", "backdoor", "shell"],
    "high": ["vulnerable", "unpatched", "outdated"],
    "medium": ["misconfigured", "weak"],
    "low": ["deprecated", "legacy"]
  }
}
```

### Descriptor Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Tool name (must be unique) |
| `description` | Yes | What the tool does |
| `category` | Yes | Tool category: `reconnaissance`, `vulnerability_scanning`, `exploitation`, `analysis` |
| `phase` | Yes | Which assessment phases use this tool (array) |
| `command` | Yes | Command to execute the tool |
| `required` | No | Whether tool must be present (default: false) |
| `args` | Yes | Command arguments for different scan profiles |
| `default_profile` | Yes | Which profile to use by default |
| `timeout` | No | Command timeout in seconds (default: 120) |
| `output_format` | No | Output format: `text`, `json`, `xml` |
| `parse_output` | No | Whether to parse output for findings (default: true) |
| `severity_keywords` | No | Keywords to match for finding severity |

### Argument Templates

Use these placeholders in `args`:
- `{target}` - Target URL/hostname
- `{port}` - Specific port
- `{output}` - Output file path

## Adding a New Tool

### Quick Method: Use the Generator Script

```bash
# Interactive mode - step-by-step prompts
python3 create_tool_descriptor.py interactive

# Auto mode - AI analyzes tool (future feature)
python3 create_tool_descriptor.py auto <tool_name>

# From help - Generate from tool's --help output
python3 create_tool_descriptor.py from-help <tool_name>
```

### Manual Method: Create Tool Descriptor

Create `tool_descriptors/<tool_name>.json`:

```json
{
  "name": "mytool",
  "description": "Description of what it does",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "mytool",
  "args": {
    "default": "-v {target}"
  },
  "default_profile": "default",
  "timeout": 60
}
```

### Step 2: Ensure Tool is Available

Make sure the tool is installed and available in PATH:

```bash
which mytool
# or install it:
# apt-get install mytool
# or add to tool_scripts/
```

### Step 3: Test Tool

The Security Agent will automatically discover and use your tool:

```bash
python3 security_agent.py example.com
```

## Example Tools

### Reconnaissance Tools
- **nmap** - Port scanning, service detection
- **nikto** - Web server scanner
- **dirb/gobuster** - Directory brute-forcing
- **whatweb** - Website fingerprinting
- **theharvester** - Email/subdomain harvesting
- **dnsenum** - DNS enumeration
- **wafw00f** - WAF detection

### Vulnerability Scanning
- **nuclei** - Fast vulnerability scanner
- **sqlmap** - SQL injection detection
- **xsstrike** - XSS vulnerability scanner
- **wpscan** - WordPress vulnerability scanner
- **testssl.sh** - SSL/TLS testing

### Analysis Tools
- **metasploit** - Exploitation framework (use responsibly!)
- **burpsuite** - Web proxy/scanner
- **zap** - OWASP ZAP proxy

## Custom Tool Scripts

For tools that need wrapper scripts, place them in `tool_scripts/`:

```bash
#!/bin/bash
# tool_scripts/custom_recon.sh

TARGET=$1

echo "Running custom reconnaissance on $TARGET"

# Run multiple tools
nslookup $TARGET
whois $TARGET | head -20
curl -sI https://$TARGET | grep -i "server\|x-"

echo "Custom recon complete"
```

Then create descriptor:

```json
{
  "name": "custom_recon",
  "description": "Custom reconnaissance script",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "bash",
  "args": {
    "default": "tool_scripts/custom_recon.sh {target}"
  },
  "default_profile": "default"
}
```

## Security Considerations

⚠️ **IMPORTANT - Authorized Testing Only**

- Only test systems you own or have explicit written permission to test
- Unauthorized security testing is illegal
- Always use the least intrusive tools first
- Avoid aggressive scans on production systems
- Follow responsible disclosure practices
- Document all testing authorization

## Tool Categories

### Reconnaissance (Phase 1)
Passive and active information gathering:
- DNS enumeration
- Port scanning
- Service fingerprinting
- Directory enumeration
- SSL/TLS analysis

### Vulnerability Scanning (Phase 2)
Identifying potential vulnerabilities:
- Web application scanning
- Network vulnerability scanning
- Configuration auditing
- Credential testing (authorized only)

### Analysis (Phase 3)
Deep analysis and verification:
- Manual verification
- False positive filtering
- Evidence gathering
- Impact assessment

## Profiles

Each tool can have multiple profiles for different scenarios:

- **basic** - Quick, non-intrusive scan (default)
- **aggressive** - Comprehensive, more intrusive scan
- **stealth** - Slow, stealthy scan to avoid detection
- **targeted** - Focused on specific vulnerabilities

## Output Parsing

The Security Agent will automatically parse tool outputs based on:

1. **Exit codes** - Non-zero usually indicates findings
2. **Severity keywords** - Matched against output text
3. **Output format** - Structured formats (JSON/XML) parsed automatically
4. **Pattern matching** - CVE IDs, version numbers, etc.

## Best Practices

1. **Start Light** - Begin with non-intrusive reconnaissance
2. **Progressively Deeper** - Gradually increase scan intensity
3. **Document Everything** - Keep logs of all scan activities
4. **Verify Findings** - Manually verify automated findings
5. **Use AI Analysis** - Let the agent reason about findings
6. **Follow Up** - Investigate interesting findings deeply

## Examples

### Minimal Tool (curl-based header check)

```json
{
  "name": "header_check",
  "description": "Check HTTP security headers",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "curl",
  "args": {
    "default": "-sI {target}"
  },
  "default_profile": "default",
  "timeout": 10,
  "severity_keywords": {
    "medium": ["no.*security.*header"]
  }
}
```

### Advanced Tool (nmap with XML output)

```json
{
  "name": "nmap_xml",
  "description": "Nmap with structured XML output",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "nmap",
  "args": {
    "default": "-sV -T4 -oX {output} {target}"
  },
  "default_profile": "default",
  "timeout": 300,
  "output_format": "xml",
  "parse_output": true
}
```

## Contributing Tools

To contribute new tool descriptors:

1. Create the descriptor JSON file
2. Test it thoroughly on authorized targets
3. Document any special requirements
4. Add usage examples
5. Submit PR or share with the team

## Support

For questions or issues with tool integration:
- Check tool is installed: `which <tool>`
- Verify JSON syntax: `python3 -m json.tool <descriptor>.json`
- Test tool manually first
- Check Security Agent logs for errors

---

**Remember: With great power comes great responsibility. Use these tools ethically and legally.**
