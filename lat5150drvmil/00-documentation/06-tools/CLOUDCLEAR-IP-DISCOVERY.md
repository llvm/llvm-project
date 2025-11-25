# CLOUDCLEAR - DNS Reconnaissance & Origin Discovery

**Project**: CLOUDCLEAR
**Repository**: https://github.com/SWORDIntel/CLOUDCLEAR
**Organization**: SWORD Intelligence (SWORDIntel)
**Category**: Network Reconnaissance / Threat Intelligence
**License**: Proprietary

![CLOUDCLEAR](https://img.shields.io/badge/CLOUDCLEAR-Origin%20Discovery-red)
![SWORD Intelligence](https://img.shields.io/badge/SWORD-Intelligence-blue)
![Authorized Use Only](https://img.shields.io/badge/Status-AUTHORIZED%20USE%20ONLY-orange)

---

## ⚠️ CRITICAL LEGAL NOTICE

**AUTHORIZED USE ONLY**: This tool is designed for **authorized security testing, threat intelligence, and defensive security operations**. Unauthorized reconnaissance, doxxing, stalking, or targeting of individuals or organizations is **ILLEGAL** and **UNETHICAL**.

**Legal Requirements**:
- ✅ Written authorization from target organization
- ✅ Penetration testing engagement contract
- ✅ Red team exercise authorization
- ✅ Threat intelligence research (defensive)
- ✅ Incident response investigation
- ✅ CTF competition or training environment

**Prohibited Uses**:
- ❌ Unauthorized reconnaissance of individuals or organizations
- ❌ Doxxing or harassment
- ❌ Bypassing security for malicious purposes
- ❌ Targeting without explicit written permission
- ❌ Stalking or surveillance
- ❌ Any activity violating CFAA, GDPR, or local laws

**Violating these restrictions may result in criminal prosecution under 18 U.S.C. § 1030 (Computer Fraud and Abuse Act) and equivalent international laws.**

---

## Executive Summary

**CLOUDCLEAR** is a next-generation DNS reconnaissance and origin discovery platform developed by SWORD Intelligence to reveal infrastructure hidden behind content delivery networks (CDNs) such as Cloudflare, Akamai, and CloudFront. It employs **9 distinct discovery vectors** to identify true origin IP addresses through correlation techniques rather than traditional DNS lookups.

**Use Cases for LAT5150DRVMIL**:
- **Threat Intelligence**: Map threat actor infrastructure hidden behind CDNs
- **Incident Response**: Locate C2 servers using Cloudflare as anonymization
- **Red Team Operations**: Authorized penetration testing to bypass CDN protections
- **Vulnerability Research**: Identify exposed origin servers for security assessment
- **APT Tracking**: Discover real infrastructure of nation-state actors

**Success Rates** (per documentation):
- **60-80%** on properly configured CDNs
- **95%+** on misconfigured infrastructure
- **70-95%** WAF bypass rate (depending on vendor)

---

## What is CLOUDCLEAR?

### The Problem: CDN Anonymization

**Challenge**: Modern web infrastructure hides origin servers behind CDN layers:

```
User Request → CDN Edge (Cloudflare) → Origin Server (Hidden)
              203.0.113.50           10.0.1.100 (Unknown)
```

**Why This Matters**:
- **Security**: Direct access to origin bypasses CDN protections (WAF, DDoS, rate limiting)
- **Threat Intelligence**: Adversaries use CDNs to hide C2 infrastructure
- **Cost**: Some attacks (DDoS amplification) work better against origins
- **Research**: Understanding real infrastructure topology

### The Solution: Multi-Vector Correlation

CLOUDCLEAR doesn't rely on single-point failures (DNS lookups). Instead, it correlates **9 independent signals** to probabilistically identify origins:

```
SSL Cert SANs + Historical DNS + Subdomain Analysis + MX Records + ...
→ High-confidence origin IP identification
```

---

## Technical Architecture

### 9 Discovery Vectors

#### 1. SSL Certificate Analysis
**Method**: Match Subject Alternative Names (SANs) across certificates

**How It Works**:
```bash
# Cloudflare edge server (CDN)
Certificate for example.com:
  SANs: example.com, www.example.com, api.example.com

# Scan IPv4 space for servers with same cert
for ip in <ip_range>; do
  cert=$(openssl s_client -connect $ip:443 -servername example.com)
  if [[ $cert contains "api.example.com" ]]; then
    echo "Potential origin: $ip"
  fi
done
```

**Why Effective**: Origin servers often use the same SSL certificate as CDN edge servers

#### 2. Historical DNS Records
**Method**: Analyze archived DNS data pre-CDN migration

**Sources**:
- SecurityTrails historical records
- DNS Dumpster archives
- Archive.org DNS snapshots
- PassiveTotal historical DNS

**Example**:
```
2020-01-15: example.com → 192.0.2.100 (Direct IP)
2021-06-20: example.com → 203.0.113.50 (Cloudflare)

Origin likely still: 192.0.2.100
```

#### 3. Direct IP Connection Testing
**Method**: Attempt connections with host header manipulation

**Technique**:
```bash
# Try connecting directly to suspected IP
curl -H "Host: example.com" http://192.0.2.100/

# If server responds with expected content → Origin found
# If 403/404/timeout → Wrong IP or protected
```

**Variations**:
- HTTP/1.1 Host header
- HTTP/2 :authority pseudo-header
- SNI (Server Name Indication) in TLS handshake

#### 4. Subdomain Correlation
**Method**: Cluster related subdomains to infer shared infrastructure

**Logic**:
```
www.example.com → 203.0.113.50 (Cloudflare)
api.example.com → 203.0.113.50 (Cloudflare)
mail.example.com → 192.0.2.100 (Direct IP) ← Origin candidate!
vpn.example.com → 192.0.2.101 (Direct IP) ← Same /24 subnet
```

**Assumption**: Mail/VPN/Internal subdomains often share hosting with web servers

#### 5. MX Record Analysis
**Method**: Identify mail servers that may share hosting with web services

**Rationale**:
```
example.com MX records:
  10 mail.example.com → 192.0.2.100

Hypothesis: Web server might be 192.0.2.100 or nearby IP
```

**Why This Works**: Small organizations often host web + mail on same server or subnet

#### 6. SRV Record Discovery
**Method**: Locate service records revealing backend infrastructure

**Common SRV Records**:
```
_caldav._tcp.example.com → caldav.example.com → 192.0.2.100
_xmpp._tcp.example.com → xmpp.example.com → 192.0.2.101
_sip._tls.example.com → sip.example.com → 192.0.2.102
```

**Application**: Backend services often bypass CDN

#### 7. Reverse PTR Lookups
**Method**: Map IP addresses to hostnames for infrastructure identification

**Process**:
```bash
# Find IPs in target ASN
whois example.com | grep "NetRange"
# NetRange: 192.0.2.0 - 192.0.2.255

# Reverse lookup each IP
for ip in 192.0.2.{1..255}; do
  ptr=$(dig -x $ip +short)
  echo "$ip → $ptr"
done

# Look for patterns:
# 192.0.2.100 → web01.example.com
# 192.0.2.101 → web02.example.com
```

#### 8. ASN/BGP Analysis
**Method**: Examine autonomous system numbers and routing data

**Sources**:
- RIPE stat
- Hurricane Electric BGP Toolkit
- ARIN WHOIS
- Team Cymru IP-to-ASN

**Example**:
```
example.com → Cloudflare (AS13335)

Historical ASN:
  AS64512 (Example Corp) owns 192.0.2.0/24

Hypothesis: Origin likely in 192.0.2.0/24
```

#### 9. IPv4/IPv6 Dual-Stack Discovery
**Method**: Identify both protocol versions for targets

**Rationale**:
```
IPv4: example.com → 203.0.113.50 (Cloudflare)
IPv6: example.com → 2001:db8::1 (Direct IPv6, no CDN!)

Origin found: 2001:db8::1
```

**Why Effective**: Many admins forget to CDN-protect IPv6

---

## DNS Protocol Support

### Encrypted DNS

**Supported Protocols**:
1. **DNS-over-HTTPS (DoH)**: RFC 8484
   - Cloudflare: https://1.1.1.1/dns-query
   - Google: https://dns.google/dns-query
   - Quad9: https://dns.quad9.net/dns-query

2. **DNS-over-TLS (DoT)**: RFC 7858
   - Port 853, TLS encryption
   - `dig @1.1.1.1 +tls example.com`

3. **DNS-over-QUIC (DoQ)**: RFC 9250
   - Next-gen encrypted DNS
   - Lower latency than DoH/DoT

4. **Traditional UDP/TCP**: Fallback with DNSSEC validation

**Benefits**:
- Bypass ISP DNS filtering
- Avoid DNS cache poisoning
- Privacy-preserving reconnaissance
- Evade DNS-based monitoring

---

## WAF Evasion Capabilities

### WAF Detection

**Fingerprinting**: Identifies 10+ major WAF vendors

**Supported WAFs**:
- Cloudflare
- Akamai Kona Site Defender
- AWS WAF
- Imperva (Incapsula)
- F5 BIG-IP ASM
- Barracuda WAF
- Fortinet FortiWeb
- ModSecurity
- Sucuri
- Wordfence

**Detection Method**:
```python
# Fingerprint WAF from response headers
headers = {
    'CF-RAY': 'Cloudflare',
    'X-CDN': 'Incapsula',
    'X-Akamai-Transformed': 'Akamai',
    'X-Sucuri-ID': 'Sucuri'
}
```

### Bypass Techniques

#### 1. Header Spoofing

**X-Forwarded-For Manipulation**:
```http
X-Forwarded-For: 127.0.0.1
X-Originating-IP: 127.0.0.1
X-Remote-IP: 127.0.0.1
X-Remote-Addr: 127.0.0.1
X-Client-IP: 127.0.0.1
CF-Connecting-IP: 127.0.0.1
True-Client-IP: 127.0.0.1
```

**Why This Works**: WAFs may whitelist localhost/internal IPs

#### 2. Chunked Transfer Encoding

**Method**: Fragment requests to confuse WAF parsers

```http
POST /api/endpoint HTTP/1.1
Transfer-Encoding: chunked

5
<?xml
4
 ver
3
sio
...
```

**Effectiveness**: Some WAFs fail to reassemble chunked requests

#### 3. HTTP Parameter Pollution

**Technique**: Duplicate parameters with different values

```
/api?id=1&id=<script>alert(1)</script>
```

**Confusion**: WAFs and backends may parse differently
- WAF sees: `id=1`
- Backend sees: `id=<script>alert(1)</script>`

#### 4. Header Case Mutation

**Method**: Vary HTTP header capitalization

```http
cOnTeNt-tYpE: application/json
UsEr-AgEnT: Mozilla/5.0
```

**Why**: Case-insensitive parsers may differ between WAF and origin

#### 5. Encoding Variations

**Multiple Encoding Layers**:
```
Original: <script>
URL:      %3Cscript%3E
Double:   %253Cscript%253E
Unicode:  \u003Cscript\u003E
Hex:      &#x3C;script&#x3E;
```

**Success Rate**: 70-95% bypass depending on WAF type

---

## Additional Reconnaissance Modules

### Zone Transfer Testing (AXFR/IXFR)

**Exploit**: Misconfigured DNS servers allowing full zone dumps

```bash
dig @ns1.example.com example.com AXFR

# If successful:
example.com.        86400   IN      A       192.0.2.100
www.example.com.    86400   IN      A       192.0.2.100
mail.example.com.   86400   IN      A       192.0.2.101
admin.example.com.  86400   IN      A       192.0.2.102  ← Hidden subdomain!
```

**Impact**: Reveals entire DNS structure including hidden subdomains

### Subdomain Brute-Force

**Performance**: 1000+ queries/second (multi-threaded)

**Wordlists**:
- SecLists DNS
- Sublist3r
- Amass wordlists
- Custom domain-specific terms

**Techniques**:
- Recursive subdomain discovery
- Permutation generation (dev-, staging-, prod-, etc.)
- Numeric iteration (web1, web2, web3...)

### HTTP Banner Grabbing

**Collected Metadata**:
- Server header (Apache, nginx, IIS version)
- X-Powered-By (PHP, ASP.NET version)
- SSL/TLS certificate details
- Technology fingerprints (WordPress, Joomla, etc.)

**Example**:
```http
HTTP/1.1 200 OK
Server: nginx/1.18.0
X-Powered-By: PHP/7.4.3
X-Frame-Options: SAMEORIGIN

# Reveals: nginx 1.18.0 (known CVEs) + PHP 7.4.3
```

### Port Scanning

**Protocols**: TCP/UDP service discovery

**Common Targets**:
- 22 (SSH) - Check for exposed management
- 3306 (MySQL) - Database direct access
- 6379 (Redis) - Cache server exposure
- 8080/8443 (Alt HTTP) - Dev/staging servers
- 27017 (MongoDB) - NoSQL databases

---

## OPSEC Features

### Traffic Randomization

**Timing Jitter**:
```python
import random
import time

for target in targets:
    scan(target)
    # Random delay: 0.5-3.0 seconds
    time.sleep(random.uniform(0.5, 3.0))
```

**Why**: Evades rate limiting and IDS temporal pattern detection

### User-Agent Rotation

**Pool**: 100+ legitimate browser user-agents

```
Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15
Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0
...
```

**Rotation**: Per-request or per-target randomization

### Rate Limiting

**Configurable Throttling**:
- Max requests per second
- Max concurrent connections
- Automatic backoff on 429 responses

### Proxy Support

**Protocols**:
- SOCKS5
- HTTP/HTTPS
- Tor integration

**Use Case**: Route traffic through multiple exit nodes for anonymity

### Auto-Pause on Detection

**Triggers**:
- HTTP 429 (Too Many Requests)
- CAPTCHA challenges
- Cloudflare "Checking your browser" pages
- TCP connection resets

**Action**: Exponential backoff + proxy rotation

---

## Installation & Usage

### Docker Deployment (Recommended)

```bash
# Pull official image
docker pull ghcr.io/swordintel/cloudclear:latest

# Run with interactive TUI
docker run -it --rm ghcr.io/swordintel/cloudclear:latest --target example.com

# Run with JSON output
docker run -it --rm ghcr.io/swordintel/cloudclear:latest --target example.com --output json > results.json
```

### Native Linux Binary

```bash
# Download latest release
wget https://github.com/SWORDIntel/CLOUDCLEAR/releases/latest/download/cloudclear-linux-amd64

# Make executable
chmod +x cloudclear-linux-amd64

# Run
./cloudclear-linux-amd64 --target example.com
```

### From Source

```bash
# Clone repository
git clone https://github.com/SWORDIntel/CLOUDCLEAR.git
cd CLOUDCLEAR

# Install dependencies
pip install -r requirements.txt

# Run
python cloudclear.py --target example.com
```

---

## Integration with LAT5150DRVMIL

### 1. Threat Intelligence Pipeline

**Use Case**: Map APT infrastructure hidden behind Cloudflare

```bash
# Discover origin IPs for known C2 domains
cloudclear --target apt29-c2.example.com --aggressive

# Results:
# Origin IP: 192.0.2.100
# ASN: AS64512 (Malicious Hosting LLC)
# Country: Russia
# Confidence: 87%

# Add to SWORD Intelligence feed
python -c "
from rag_system.cerebras_integration import CerebrasCloud
cerebras = CerebrasCloud()
analysis = cerebras.threat_intelligence_query('APT29 infrastructure 192.0.2.100')
print(analysis)
"
```

### 2. Malware C2 Analysis

**Workflow**:
```python
# Extract C2 domain from malware sample
from rag_system.neural_code_synthesis import NeuralCodeSynthesizer

synthesizer = NeuralCodeSynthesizer(rag_retriever=None)
analyzer = synthesizer.generate_module("Malware analyzer with IOC extraction")

# Analyzer extracts: c2-server.evil.com
# Run CLOUDCLEAR to find origin
# → 198.51.100.50 (Real C2 IP)

# Block at firewall
subprocess.run(['iptables', '-A', 'OUTPUT', '-d', '198.51.100.50', '-j', 'DROP'])
```

### 3. Red Team Operations

**Scenario**: Authorized penetration test, bypass CDN to access origin

```bash
# 1. Discover origin IP
cloudclear --target client-website.com

# 2. Verify origin responds
curl -H "Host: client-website.com" http://203.0.113.100/

# 3. Test for vulnerabilities on origin (bypassing WAF)
nmap -sV -p- 203.0.113.100
nikto -h http://203.0.113.100/ -vhost client-website.com

# 4. Report findings to client
```

### 4. Incident Response

**Scenario**: Identify attacker infrastructure during active incident

```bash
# Attacker domain: phishing-site.badactor.com (behind Cloudflare)

# Find real IP
cloudclear --target phishing-site.badactor.com --fast

# Origin: 192.0.2.200
# ASN: AS64513 (BulletProof Hosting)

# Coordinate takedown with hosting provider
# Document for law enforcement
```

### 5. Forensics & Attribution

**Use Case**: Track threat actor across infrastructure changes

```bash
# Historical analysis
cloudclear --target attacker-site.com --historical

# Results:
# 2023-01-15: 192.0.2.10 (OVH)
# 2023-06-20: 192.0.2.20 (DigitalOcean)
# 2024-01-10: 192.0.2.30 (Vultr)

# Pattern: Moves hosting every 6 months
# All IPs in same /16 subnet (192.0.2.0/16)
# Attribution: Same threat actor
```

---

## Output Formats

### Interactive TUI (Terminal UI)

```
┌─ CLOUDCLEAR - Origin Discovery ─────────────────────────┐
│ Target: example.com                                      │
│ Status: Scanning... [9/9 vectors complete]              │
├──────────────────────────────────────────────────────────┤
│ ✓ SSL Certificate Analysis    [3 candidates]            │
│ ✓ Historical DNS Records       [1 match]                │
│ ✓ Direct IP Testing            [1 confirmed]            │
│ ✓ Subdomain Correlation        [2 related]              │
│ ✓ MX Record Analysis           [1 candidate]            │
│ ✓ SRV Record Discovery         [0 found]                │
│ ✓ Reverse PTR Lookups          [1 match]                │
│ ✓ ASN/BGP Analysis             [AS64512]                │
│ ✓ IPv6 Dual-Stack              [1 IPv6 found]           │
├──────────────────────────────────────────────────────────┤
│ HIGH CONFIDENCE ORIGIN:                                  │
│   IP: 192.0.2.100                                        │
│   Confidence: 94%                                        │
│   Verified: Yes (HTTP 200 with correct Host header)     │
│   ASN: AS64512 (Example Corp)                            │
│   Country: United States                                 │
│   Technologies: nginx/1.18.0, PHP/7.4.3                  │
└──────────────────────────────────────────────────────────┘
```

### JSON Output

```json
{
  "target": "example.com",
  "scan_timestamp": "2025-11-08T12:34:56Z",
  "cdn_detected": "Cloudflare",
  "origin_ips": [
    {
      "ip": "192.0.2.100",
      "confidence": 94,
      "verified": true,
      "asn": "AS64512",
      "country": "US",
      "sources": [
        "ssl_certificate",
        "historical_dns",
        "direct_connection",
        "ptr_lookup"
      ],
      "technologies": {
        "server": "nginx/1.18.0",
        "language": "PHP/7.4.3"
      }
    }
  ],
  "ipv6_addresses": [
    {
      "ip": "2001:db8::1",
      "confidence": 87,
      "verified": true
    }
  ],
  "subdomains": [
    "mail.example.com",
    "vpn.example.com",
    "admin.example.com"
  ],
  "mx_records": [
    {
      "priority": 10,
      "hostname": "mail.example.com",
      "ip": "192.0.2.101"
    }
  ]
}
```

### CSV Output

```csv
target,origin_ip,confidence,verified,asn,country,server,sources
example.com,192.0.2.100,94,true,AS64512,US,nginx/1.18.0,"ssl_certificate,historical_dns,direct_connection"
example.com,2001:db8::1,87,true,AS64512,US,nginx/1.18.0,"ipv6_dual_stack"
```

---

## Defensive Countermeasures

### How to Protect Your Origin

If you manage infrastructure behind CDNs, CLOUDCLEAR reveals weaknesses. Here's how to harden:

#### 1. Firewall Rules (Origin Protection)

**Block All Direct Access**:
```bash
# iptables: Only allow Cloudflare IPs
iptables -A INPUT -p tcp --dport 80 -s 173.245.48.0/20 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -s 173.245.48.0/20 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j DROP
iptables -A INPUT -p tcp --dport 443 -j DROP
```

**Cloudflare IP Ranges**: https://www.cloudflare.com/ips/

#### 2. Remove Historical DNS Records

**Problem**: Archive.org, SecurityTrails have old IPs

**Solution**:
- Change origin IP after CDN migration
- Use different IP ranges pre/post-CDN
- Request removal from DNS archives (limited effectiveness)

#### 3. Separate Infrastructure

**Don't Share IPs**:
- Mail server: Different IP/subnet than web
- VPN: Separate network entirely
- Internal tools: Behind dedicated firewall

#### 4. Disable Zone Transfers

**BIND Configuration**:
```
zone "example.com" {
    type master;
    file "/etc/bind/db.example.com";
    allow-transfer { none; };  # Disable AXFR
};
```

#### 5. Consistent SSL Certificates

**Problem**: Same cert on origin and CDN edge

**Solution**:
- Use different certs for origin vs edge
- Wildcard cert only on edge
- Origin cert with restricted SANs

#### 6. IPv6 Protection

**Don't forget IPv6**:
```bash
# Ensure IPv6 also behind CDN
example.com AAAA 2606:4700:xxxx (Cloudflare IPv6)

# Block direct IPv6 to origin
ip6tables -A INPUT -p tcp --dport 80 -j DROP
ip6tables -A INPUT -p tcp --dport 443 -j DROP
```

#### 7. Authenticated Origin Pulls

**Cloudflare Authenticated Origin Pulls**:
```nginx
# nginx: Require Cloudflare client certificate
ssl_client_certificate /path/to/origin-pull-ca.pem;
ssl_verify_client on;
```

**Why**: Even if IP is discovered, direct connections fail without Cloudflare client cert

---

## Ethical and Legal Considerations

### Authorized Use Cases

**Legitimate Applications**:

1. **Penetration Testing**: Written contract with explicit scope
2. **Bug Bounty Programs**: Within program rules
3. **Incident Response**: Active threat investigation
4. **Threat Intelligence**: Mapping adversary infrastructure
5. **Red Team Exercises**: Internal security testing
6. **Security Research**: Academic or defensive purposes
7. **CTF Competitions**: Authorized training environments

**Documentation Required**:
- Signed engagement letter
- Scope of work (target list)
- Timeline and constraints
- Legal liability waivers

### Prohibited Use Cases

**Illegal Activities**:

1. **Unauthorized Reconnaissance**: Scanning targets without permission
2. **Doxxing**: Finding personal infrastructure for harassment
3. **Stalking**: Tracking individuals without consent
4. **DDoS Preparation**: Identifying targets for attacks
5. **Credential Stuffing**: Finding login portals to brute-force
6. **Data Exfiltration**: Accessing unprotected origins for data theft

**Legal Consequences**:
- **CFAA (18 U.S.C. § 1030)**: Up to 10 years imprisonment
- **GDPR Article 83**: Fines up to €20 million
- **UK Computer Misuse Act**: Up to 2 years imprisonment
- Civil liability for damages

### Responsible Disclosure

**If you discover exposed origins**:

1. **Notify the organization** (security@ email)
2. **Provide technical details** (IP, discovery method)
3. **Give reasonable timeline** (30-90 days)
4. **Don't publicly disclose** until patched
5. **Don't exploit vulnerability** beyond proof-of-concept

---

## Integration with SWORD Intelligence

### Threat Actor Infrastructure Mapping

```python
# Example: Track APT29 infrastructure
from rag_system.cerebras_integration import CerebrasCloud

cerebras = CerebrasCloud()

# CLOUDCLEAR discovers: 198.51.100.50
# Query Cerebras for attribution
analysis = cerebras.threat_intelligence_query(
    """
    APT29 infrastructure analysis:
    - IP: 198.51.100.50
    - ASN: AS64513 (Russian hosting)
    - Hosting history: OVH → DigitalOcean → Vultr
    - SSL cert CN: *.cozy-bear.ru
    """
)

print(analysis['analysis'])
# Output: "High confidence APT29 (Cozy Bear) based on:
# 1. ASN matches known Russian APT hosting patterns
# 2. Infrastructure rotation consistent with APT29 TTPs
# 3. SSL cert naming convention matches MITRE ATT&CK data
# Recommend: Block at perimeter, monitor for lateral movement"
```

### Automated IOC Extraction

```python
# Generate YARA rule for discovered infrastructure
yara_rule = cerebras.generate_yara_rule(
    """
    APT29 C2 server discovered via CLOUDCLEAR:
    - Origin IP: 198.51.100.50
    - Domain: state-dept-secure.com (typosquat)
    - SSL cert fingerprint: AA:BB:CC:DD...
    """
)

print(yara_rule)
# Outputs production YARA rule for detection
```

---

## Performance Benchmarks

### Scan Speed

**Single Target** (all 9 vectors):
- Fast mode: 30-60 seconds
- Standard mode: 2-5 minutes
- Aggressive mode: 10-20 minutes

**Bulk Scanning** (1000 domains):
- Multi-threaded: 20-50 domains/minute
- Total time: ~20-50 minutes

### Resource Usage

**CPU**: 2-4 cores (multi-threaded)
**RAM**: 512 MB - 2 GB (depending on wordlist size)
**Network**: 1-10 Mbps (throttled to avoid detection)

### Success Rates

**CDN Type**:
- Cloudflare: 65-75%
- Akamai: 70-80%
- AWS CloudFront: 60-70%
- Fastly: 55-65%
- Misconfigured (any): 95%+

**Confidence Levels**:
- 90-100%: Very High (4+ vector confirmation)
- 75-89%: High (3 vector confirmation)
- 50-74%: Medium (2 vector confirmation)
- <50%: Low (1 vector, requires verification)

---

## Troubleshooting

### Issue 1: All Vectors Fail

**Symptoms**: No origin IPs discovered, 0% confidence

**Causes**:
1. Proper CDN configuration (origin fully protected)
2. Rate limiting triggered
3. Network connectivity issues

**Solutions**:
```bash
# Reduce scan speed
cloudclear --target example.com --delay 2.0 --max-threads 2

# Use proxy rotation
cloudclear --target example.com --proxy socks5://127.0.0.1:9050

# Try IPv6 only
cloudclear --target example.com --ipv6-only
```

### Issue 2: False Positives

**Symptoms**: High confidence but wrong IP

**Verification**:
```bash
# Manual verification
curl -H "Host: example.com" http://DISCOVERED_IP/

# Check for expected content
# Compare with legitimate site
```

### Issue 3: Cloudflare Detection

**Symptoms**: "Checking your browser" CAPTCHA pages

**Solutions**:
```bash
# Rotate user-agents more frequently
cloudclear --target example.com --ua-rotate per-request

# Increase jitter
cloudclear --target example.com --jitter 5.0

# Use residential proxies (not datacenter IPs)
```

---

## References

### Official Resources
- **CLOUDCLEAR Repository**: https://github.com/SWORDIntel/CLOUDCLEAR
- **SWORD Intelligence**: https://github.com/SWORDOps/SWORDINTELLIGENCE/

### Related Tools
- **CloudFlair**: Historical approach (less sophisticated)
- **CloudScraper**: WAF bypass focus
- **CrimeFlare**: Community database of origins
- **Shodan/Censys**: Certificate transparency logs

### Academic Research
- *"Bypassing CDN Protection"* - Black Hat USA 2017
- *"Origin IP Discovery via Certificate Correlation"* - DEF CON 26
- *"DNS Archaeology for Offensive Operations"* - OffensiveCon 2019

### Legal Framework
- **CFAA**: 18 U.S.C. § 1030
- **GDPR**: Article 6 (lawful processing)
- **NIST Cybersecurity Framework**: RS.AN (Analysis)

---

## Conclusion

**CLOUDCLEAR** is a powerful reconnaissance tool that reveals infrastructure hidden behind CDN protections. When used **legally and ethically**, it serves critical functions in:

- **Defensive Security**: Understanding your own exposure
- **Threat Intelligence**: Mapping adversary infrastructure
- **Incident Response**: Tracking attacker systems
- **Penetration Testing**: Authorized security assessments

**Remember**: Power requires responsibility. Always obtain **written authorization** before scanning any targets. Unauthorized use is **illegal** and **unethical**.

For LAT5150DRVMIL operations, CLOUDCLEAR integrates seamlessly with:
- SWORD Intelligence threat feeds
- Cerebras Cloud attribution analysis
- Malware C2 infrastructure tracking
- Red team authorized engagements

---

## Document Classification

**Classification**: UNCLASSIFIED//PUBLIC
**Sensitivity**: DUAL-USE SECURITY TOOL
**Last Updated**: 2025-11-08
**Version**: 1.0
**Author**: LAT5150DRVMIL Security Research Team
**Contact**: SWORD Intelligence (https://github.com/SWORDOps/SWORDINTELLIGENCE/)

---

**FINAL WARNING**: This documentation is provided for educational and authorized security purposes only. The authors and SWORD Intelligence assume no liability for misuse. Users are solely responsible for compliance with applicable laws and regulations.

**By using CLOUDCLEAR, you acknowledge**:
1. You have explicit written authorization
2. You understand legal implications
3. You will use responsibly and ethically
4. You accept full legal responsibility for your actions
