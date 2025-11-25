#!/usr/bin/env python3
"""
Security Testing Agent

Autonomous security assessment agent inspired by Pensar APEX.
Performs multi-phase security testing: Reconnaissance → Analysis → Reporting

Features:
- Autonomous black-box security testing
- Multi-phase assessment workflow
- Integration with ACE-FCA context management
- Structured security reporting
- LOCAL-FIRST: No external API dependencies for security tools
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from tool_operations import ToolOps


@dataclass
class ToolDescriptor:
    """Security tool descriptor loaded from JSON"""
    name: str
    description: str
    category: str
    phase: List[str]
    command: str
    required: bool
    args: Dict[str, str]
    default_profile: str
    timeout: int
    output_format: str
    parse_output: bool
    severity_keywords: Dict[str, List[str]]
    available: bool = False  # Set to True if tool is in PATH

    @classmethod
    def from_json(cls, json_data: Dict) -> 'ToolDescriptor':
        """Load tool descriptor from JSON"""
        return cls(
            name=json_data["name"],
            description=json_data["description"],
            category=json_data["category"],
            phase=json_data["phase"],
            command=json_data["command"],
            required=json_data.get("required", False),
            args=json_data["args"],
            default_profile=json_data["default_profile"],
            timeout=json_data.get("timeout", 120),
            output_format=json_data.get("output_format", "text"),
            parse_output=json_data.get("parse_output", True),
            severity_keywords=json_data.get("severity_keywords", {})
        )


class ToolRegistry:
    """
    Registry for security tools

    Enumerates and manages security tools from tool_descriptors directory
    """

    def __init__(self, tools_dir: Optional[Path] = None):
        """
        Initialize tool registry

        Args:
            tools_dir: Path to security_tools directory
        """
        if tools_dir is None:
            tools_dir = Path(__file__).parent / "security_tools"

        self.tools_dir = Path(tools_dir)
        self.descriptors_dir = self.tools_dir / "tool_descriptors"
        self.tools: Dict[str, ToolDescriptor] = {}
        self._enumerate_tools()

    def _enumerate_tools(self):
        """Enumerate all tools from descriptors directory"""
        if not self.descriptors_dir.exists():
            print(f"⚠️  Tool descriptors directory not found: {self.descriptors_dir}")
            return

        # Load all JSON descriptors
        for json_file in self.descriptors_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    descriptor = ToolDescriptor.from_json(data)

                    # Check if tool is available in PATH
                    descriptor.available = self._check_tool_available(descriptor.command)

                    self.tools[descriptor.name] = descriptor

            except Exception as e:
                print(f"⚠️  Failed to load tool descriptor {json_file.name}: {e}")

    def _check_tool_available(self, command: str) -> bool:
        """Check if tool command is available in PATH"""
        try:
            result = subprocess.run(
                ["which", command],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def get_tools_for_phase(self, phase: str, only_available: bool = True) -> List[ToolDescriptor]:
        """
        Get tools suitable for a specific phase

        Args:
            phase: Phase name (e.g., "reconnaissance", "analysis")
            only_available: Only return tools that are available (default: True)

        Returns:
            List of ToolDescriptor objects
        """
        tools = []
        for tool in self.tools.values():
            if phase in tool.phase:
                if only_available and not tool.available:
                    continue
                tools.append(tool)
        return tools

    def get_tool(self, name: str) -> Optional[ToolDescriptor]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self, only_available: bool = False) -> List[str]:
        """List all tool names"""
        if only_available:
            return [name for name, tool in self.tools.items() if tool.available]
        return list(self.tools.keys())

    def print_status(self):
        """Print status of all tools"""
        print("\n🔧 Security Tools Registry")
        print("=" * 70)

        total = len(self.tools)
        available = sum(1 for t in self.tools.values() if t.available)

        print(f"Total tools: {total}")
        print(f"Available: {available}")
        print(f"Missing: {total - available}\n")

        # Group by category
        by_category = {}
        for tool in self.tools.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)

        for category, tools in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            for tool in sorted(tools, key=lambda t: t.name):
                status = "✅" if tool.available else "❌"
                print(f"  {status} {tool.name:15s} - {tool.description}")


class SecurityPhase(Enum):
    """Security assessment phases"""
    RECONNAISSANCE = "reconnaissance"
    ANALYSIS = "analysis"
    REPORTING = "reporting"


class SeverityLevel(Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityFinding:
    """Individual security finding"""
    title: str
    description: str
    severity: SeverityLevel
    category: str  # e.g., "ssl", "headers", "ports", "dns"
    evidence: str
    recommendation: str
    cve_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "cve_ids": self.cve_ids
        }


@dataclass
class SecurityReport:
    """Complete security assessment report"""
    target: str
    timestamp: str
    phases_completed: List[str]
    findings: List[SecurityFinding]
    summary: str
    total_findings: int
    findings_by_severity: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "phases_completed": self.phases_completed,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "total_findings": self.total_findings,
            "findings_by_severity": self.findings_by_severity
        }

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export report as JSON"""
        report_json = json.dumps(self.to_dict(), indent=2)
        if filepath:
            filepath.write_text(report_json)
        return report_json


class SecurityAgent:
    """
    Autonomous security testing agent

    Performs comprehensive security assessments using AI-driven analysis
    and standard security tools. Designed for authorized testing only.
    """

    def __init__(self,
                 ai_model: Optional[object] = None,
                 workspace_root: Optional[Path] = None,
                 enable_aggressive_scans: bool = False,
                 tools_dir: Optional[Path] = None):
        """
        Initialize security agent

        Args:
            ai_model: AI engine for autonomous analysis (optional)
            workspace_root: Working directory for tool operations
            enable_aggressive_scans: Enable more intrusive scanning
            tools_dir: Path to security_tools directory (optional)
        """
        self.ai_model = ai_model
        self.tool_ops = ToolOps(workspace_root=str(workspace_root) if workspace_root else None)
        self.enable_aggressive = enable_aggressive_scans
        self.findings: List[SecurityFinding] = []
        self.recon_data: Dict = {}

        # Initialize tool registry
        self.tool_registry = ToolRegistry(tools_dir=tools_dir)
        self.tool_results: Dict[str, Dict] = {}  # Store tool execution results

    def assess_target(self, target: str, phases: Optional[List[SecurityPhase]] = None) -> SecurityReport:
        """
        Perform complete security assessment

        Args:
            target: Target URL or hostname (e.g., "example.com", "https://example.com")
            phases: Phases to run (default: all phases)

        Returns:
            SecurityReport with findings
        """
        if phases is None:
            phases = [SecurityPhase.RECONNAISSANCE, SecurityPhase.ANALYSIS, SecurityPhase.REPORTING]

        print(f"🔒 Security Assessment Starting: {target}")
        print(f"⚠️  Authorized testing only - ensure you have permission!")
        print(f"📋 Phases: {', '.join([p.value for p in phases])}\n")

        self.findings = []
        self.recon_data = {}
        phases_completed = []

        # Phase 1: RECONNAISSANCE
        if SecurityPhase.RECONNAISSANCE in phases:
            print("🔍 Phase 1: Reconnaissance")
            self.recon_data = self._phase_reconnaissance(target)
            phases_completed.append("reconnaissance")
            print(f"   ✓ Gathered {len(self.recon_data)} data points\n")

        # Phase 2: ANALYSIS
        if SecurityPhase.ANALYSIS in phases:
            print("🔬 Phase 2: Security Analysis")
            analysis_findings = self._phase_analysis(target, self.recon_data)
            self.findings.extend(analysis_findings)
            phases_completed.append("analysis")
            print(f"   ✓ Identified {len(analysis_findings)} findings\n")

        # Phase 3: REPORTING
        if SecurityPhase.REPORTING in phases:
            print("📊 Phase 3: Report Generation")
            report = self._phase_reporting(target, phases_completed)
            phases_completed.append("reporting")
            print(f"   ✓ Report generated\n")
            return report

        # If reporting phase not requested, generate basic report
        return self._generate_basic_report(target, phases_completed)

    def _execute_tool(self, tool: ToolDescriptor, target: str, profile: Optional[str] = None) -> Dict:
        """
        Execute a security tool from the registry

        Args:
            tool: ToolDescriptor to execute
            target: Target URL/hostname
            profile: Tool profile to use (default: tool's default_profile)

        Returns:
            Dict with execution results
        """
        if profile is None:
            profile = tool.default_profile

        # Get command arguments for this profile
        args_template = tool.args.get(profile)
        if not args_template:
            return {
                "error": f"Profile '{profile}' not found for tool {tool.name}",
                "tool": tool.name
            }

        # Format arguments with target
        clean_target = target.replace("https://", "").replace("http://", "").split("/")[0]
        args = args_template.replace("{target}", clean_target)

        # Build full command
        full_command = f"{tool.command} {args}"

        print(f"   → Running {tool.name}...")

        # Execute tool
        result = self.tool_ops.bash(
            full_command,
            timeout=tool.timeout,
            description=f"{tool.name} {profile} scan"
        )

        # Store result
        self.tool_results[tool.name] = result

        return result

    def _parse_tool_output(self, tool: ToolDescriptor, output: str) -> List[SecurityFinding]:
        """
        Parse tool output for security findings using severity keywords

        Args:
            tool: ToolDescriptor that generated the output
            output: Tool output text

        Returns:
            List of SecurityFinding objects
        """
        findings = []

        if not tool.parse_output or not tool.severity_keywords:
            return findings

        # Check output against severity keywords
        for severity_str, keywords in tool.severity_keywords.items():
            try:
                severity = SeverityLevel[severity_str.upper()]
            except KeyError:
                continue

            for keyword in keywords:
                # Case-insensitive search
                if keyword.lower() in output.lower():
                    # Extract context around keyword (±100 chars)
                    idx = output.lower().find(keyword.lower())
                    context_start = max(0, idx - 100)
                    context_end = min(len(output), idx + len(keyword) + 100)
                    evidence = output[context_start:context_end].strip()

                    findings.append(SecurityFinding(
                        title=f"{tool.name}: {keyword} detected",
                        description=f"Tool {tool.name} detected potential issue: {keyword}",
                        severity=severity,
                        category=tool.category,
                        evidence=evidence[:200],  # Limit evidence length
                        recommendation=f"Review {tool.name} output for details and validate finding"
                    ))

        return findings

    def _phase_reconnaissance(self, target: str) -> Dict:
        """
        Phase 1: Reconnaissance - Gather information about target

        Uses both built-in checks and enumerated security tools

        Returns:
            Dict with reconnaissance findings
        """
        recon = {}

        # Clean target (remove protocol if present)
        clean_target = target.replace("https://", "").replace("http://", "").split("/")[0]

        # Built-in reconnaissance checks
        print("   → DNS lookup...")
        dns_result = self.tool_ops.bash(f"nslookup {clean_target} 2>&1 || dig {clean_target} +short 2>&1",
                                       timeout=10)
        if dns_result.get("success"):
            recon["dns"] = dns_result["stdout"]

        # HTTP Headers
        print("   → HTTP headers...")
        for protocol in ["https", "http"]:
            url = f"{protocol}://{clean_target}"
            headers_result = self.tool_ops.bash(
                f"curl -sI -m 10 {url} 2>&1 | head -30",
                timeout=15
            )
            if headers_result.get("success") and headers_result["stdout"]:
                recon[f"{protocol}_headers"] = headers_result["stdout"]
                break

        # SSL/TLS Certificate (if HTTPS)
        if target.startswith("https://") or not target.startswith("http://"):
            print("   → SSL/TLS certificate...")
            ssl_result = self.tool_ops.bash(
                f"echo | openssl s_client -connect {clean_target}:443 -servername {clean_target} 2>&1 | "
                f"openssl x509 -noout -text 2>&1 | head -50",
                timeout=15
            )
            if ssl_result.get("success"):
                recon["ssl_cert"] = ssl_result["stdout"]

        # WHOIS (basic domain info)
        print("   → WHOIS lookup...")
        whois_result = self.tool_ops.bash(f"whois {clean_target} 2>&1 | head -30", timeout=10)
        if whois_result.get("success"):
            recon["whois"] = whois_result["stdout"]

        # Ping (host availability)
        print("   → Host availability...")
        ping_result = self.tool_ops.bash(f"ping -c 3 {clean_target} 2>&1", timeout=10)
        if ping_result.get("success"):
            recon["ping"] = ping_result["stdout"]

        # Common security headers check
        print("   → Security headers...")
        if "https_headers" in recon or "http_headers" in recon:
            headers = recon.get("https_headers", recon.get("http_headers", ""))
            recon["security_headers"] = self._extract_security_headers(headers)

        # Execute enumerated reconnaissance tools
        recon_tools = self.tool_registry.get_tools_for_phase("reconnaissance", only_available=True)
        if recon_tools:
            print(f"   → Running {len(recon_tools)} enumerated tools...")
            for tool in recon_tools:
                try:
                    result = self._execute_tool(tool, target)
                    if result.get("success"):
                        recon[f"tool_{tool.name}"] = result["stdout"]

                        # Parse tool output for findings
                        tool_findings = self._parse_tool_output(tool, result["stdout"])
                        if tool_findings:
                            self.findings.extend(tool_findings)
                            print(f"      ✓ {tool.name}: {len(tool_findings)} findings")
                except Exception as e:
                    print(f"      ⚠️  {tool.name} failed: {e}")

        return recon

    def _phase_analysis(self, target: str, recon_data: Dict) -> List[SecurityFinding]:
        """
        Phase 2: Analysis - Analyze reconnaissance data for vulnerabilities

        Uses both built-in analysis and enumerated security tools

        Returns:
            List of SecurityFinding objects
        """
        findings = []

        # Built-in analysis
        # Analyze SSL/TLS
        if "ssl_cert" in recon_data:
            findings.extend(self._analyze_ssl(recon_data["ssl_cert"]))

        # Analyze HTTP headers
        if "security_headers" in recon_data:
            findings.extend(self._analyze_security_headers(recon_data["security_headers"]))

        # Analyze DNS
        if "dns" in recon_data:
            findings.extend(self._analyze_dns(recon_data["dns"]))

        # Execute enumerated analysis tools (vulnerability scanners)
        analysis_tools = self.tool_registry.get_tools_for_phase("analysis", only_available=True)
        if analysis_tools:
            print(f"   → Running {len(analysis_tools)} vulnerability scanning tools...")
            for tool in analysis_tools:
                try:
                    # Use basic profile for analysis (avoid aggressive scans by default)
                    profile = "basic" if "basic" in tool.args else tool.default_profile
                    result = self._execute_tool(tool, target, profile=profile)

                    if result.get("success"):
                        # Parse tool output for findings
                        tool_findings = self._parse_tool_output(tool, result["stdout"])
                        if tool_findings:
                            findings.extend(tool_findings)
                            print(f"      ✓ {tool.name}: {len(tool_findings)} findings")
                except Exception as e:
                    print(f"      ⚠️  {tool.name} failed: {e}")

        # If AI model available, use it for advanced analysis
        if self.ai_model:
            findings.extend(self._ai_analysis(target, recon_data))

        return findings

    def _phase_reporting(self, target: str, phases_completed: List[str]) -> SecurityReport:
        """
        Phase 3: Reporting - Generate structured security report

        Returns:
            SecurityReport object
        """
        # Count findings by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }

        for finding in self.findings:
            severity_counts[finding.severity.value] += 1

        # Generate summary
        summary = self._generate_summary(target, severity_counts)

        return SecurityReport(
            target=target,
            timestamp=datetime.now().isoformat(),
            phases_completed=phases_completed,
            findings=self.findings,
            summary=summary,
            total_findings=len(self.findings),
            findings_by_severity=severity_counts
        )

    def _extract_security_headers(self, headers: str) -> Dict[str, Optional[str]]:
        """Extract security-relevant HTTP headers"""
        security_headers = {
            "strict-transport-security": None,
            "x-frame-options": None,
            "x-content-type-options": None,
            "content-security-policy": None,
            "x-xss-protection": None,
            "referrer-policy": None,
            "permissions-policy": None
        }

        for line in headers.lower().split("\n"):
            for header in security_headers.keys():
                if line.startswith(header + ":"):
                    security_headers[header] = line.split(":", 1)[1].strip()

        return security_headers

    def _analyze_ssl(self, ssl_cert: str) -> List[SecurityFinding]:
        """Analyze SSL/TLS certificate for issues"""
        findings = []

        # Check for weak signature algorithms
        if "sha1" in ssl_cert.lower():
            findings.append(SecurityFinding(
                title="Weak SSL Signature Algorithm",
                description="Certificate uses SHA-1 signature algorithm which is deprecated",
                severity=SeverityLevel.MEDIUM,
                category="ssl",
                evidence=f"SHA-1 found in certificate",
                recommendation="Upgrade to SHA-256 or stronger signature algorithm"
            ))

        # Check certificate expiration (basic check)
        if "Not After" in ssl_cert:
            findings.append(SecurityFinding(
                title="SSL Certificate Expiration Info",
                description="SSL certificate expiration date found",
                severity=SeverityLevel.INFO,
                category="ssl",
                evidence=f"Certificate expiration info available",
                recommendation="Monitor certificate expiration and renew before expiry"
            ))

        return findings

    def _analyze_security_headers(self, headers: Dict[str, Optional[str]]) -> List[SecurityFinding]:
        """Analyze security headers for missing protections"""
        findings = []

        # Check for missing HSTS
        if not headers.get("strict-transport-security"):
            findings.append(SecurityFinding(
                title="Missing Strict-Transport-Security Header",
                description="HSTS header not found - site may be vulnerable to SSL stripping attacks",
                severity=SeverityLevel.MEDIUM,
                category="headers",
                evidence="Strict-Transport-Security header not present",
                recommendation="Add 'Strict-Transport-Security: max-age=31536000; includeSubDomains' header"
            ))

        # Check for missing X-Frame-Options
        if not headers.get("x-frame-options"):
            findings.append(SecurityFinding(
                title="Missing X-Frame-Options Header",
                description="Site may be vulnerable to clickjacking attacks",
                severity=SeverityLevel.MEDIUM,
                category="headers",
                evidence="X-Frame-Options header not present",
                recommendation="Add 'X-Frame-Options: DENY' or 'SAMEORIGIN' header"
            ))

        # Check for missing Content-Security-Policy
        if not headers.get("content-security-policy"):
            findings.append(SecurityFinding(
                title="Missing Content-Security-Policy Header",
                description="Site lacks CSP protection against XSS and data injection attacks",
                severity=SeverityLevel.MEDIUM,
                category="headers",
                evidence="Content-Security-Policy header not present",
                recommendation="Implement Content-Security-Policy with appropriate directives"
            ))

        # Check for missing X-Content-Type-Options
        if not headers.get("x-content-type-options"):
            findings.append(SecurityFinding(
                title="Missing X-Content-Type-Options Header",
                description="Browser may be vulnerable to MIME type sniffing attacks",
                severity=SeverityLevel.LOW,
                category="headers",
                evidence="X-Content-Type-Options header not present",
                recommendation="Add 'X-Content-Type-Options: nosniff' header"
            ))

        return findings

    def _analyze_dns(self, dns_output: str) -> List[SecurityFinding]:
        """Analyze DNS configuration"""
        findings = []

        # Check for multiple A records (load balancing/redundancy indicator)
        a_records = []
        for line in dns_output.split("\n"):
            if "Address:" in line or line.strip() and "." in line and line.split(".")[0].isdigit():
                a_records.append(line.strip())

        if len(a_records) > 1:
            findings.append(SecurityFinding(
                title="Multiple DNS A Records Detected",
                description=f"Found {len(a_records)} A records - may indicate load balancing",
                severity=SeverityLevel.INFO,
                category="dns",
                evidence=f"{len(a_records)} A records found",
                recommendation="Verify all IPs are legitimate and properly configured"
            ))

        return findings

    def _ai_analysis(self, target: str, recon_data: Dict) -> List[SecurityFinding]:
        """
        Heuristic analysis that mimics AI reasoning across recon data sources.
        """
        findings: List[SecurityFinding] = []
        headers_raw = recon_data.get("https_headers") or recon_data.get("http_headers", "")
        security_headers = recon_data.get("security_headers", {})
        clean_target = target.replace("https://", "").replace("http://", "").split("/")[0]

        # HTTPS not detected
        https_present = "https_headers" in recon_data or "ssl_cert" in recon_data
        if not https_present and recon_data.get("http_headers"):
            findings.append(SecurityFinding(
                title="HTTPS Not Enforced",
                description="Target responded to HTTP but no HTTPS endpoint was detected.",
                severity=SeverityLevel.HIGH,
                category="ssl",
                evidence="HTTPS headers/certificate were not captured during reconnaissance.",
                recommendation="Redirect HTTP to HTTPS and deploy a valid TLS certificate."
            ))

        # Server banner exposure
        server_header = self._extract_header_value(headers_raw, "server")
        if server_header:
            banner_lower = server_header.lower()
            severity = SeverityLevel.LOW
            outdated_signatures = ["apache/2.2", "iis/6.0", "php/5.", "nginx/1.12"]
            if any(sig in banner_lower for sig in outdated_signatures):
                severity = SeverityLevel.MEDIUM

            findings.append(SecurityFinding(
                title="Server Banner Reveals Stack Version",
                description="Server response discloses the software stack and version.",
                severity=severity,
                category="headers",
                evidence=server_header,
                recommendation="Hide or sanitize the Server header to reduce attacker reconnaissance."
            ))

        x_powered = self._extract_header_value(headers_raw, "x-powered-by")
        if x_powered:
            severity = SeverityLevel.LOW
            if any(sig in x_powered.lower() for sig in ["php/5", "asp.net 2", "express/4"]):
                severity = SeverityLevel.MEDIUM
            findings.append(SecurityFinding(
                title="Framework Disclosure",
                description="Application framework information is exposed via X-Powered-By header.",
                severity=severity,
                category="headers",
                evidence=x_powered,
                recommendation="Remove X-Powered-By header or upgrade to a supported runtime."
            ))

        # Advanced security headers
        extra_headers = {
            "x-xss-protection": (
                SeverityLevel.LOW,
                "Legacy X-XSS-Protection header missing; some browsers still respect it."
            ),
            "referrer-policy": (
                SeverityLevel.INFO,
                "Referrer-Policy header missing; URLs may leak sensitive paths."
            ),
            "permissions-policy": (
                SeverityLevel.INFO,
                "Permissions-Policy header missing; browser features cannot be restricted."
            )
        }
        for header, (severity, description) in extra_headers.items():
            if not security_headers.get(header):
                findings.append(SecurityFinding(
                    title=f"Missing {header.title().replace('-', ' ')} Header",
                    description=description,
                    severity=severity,
                    category="headers",
                    evidence=f"{header} header absent from HTTP response.",
                    recommendation=f"Add an explicit '{header}' header aligned with security policy."
                ))

        # Certificate host mismatch
        ssl_cert = recon_data.get("ssl_cert")
        if ssl_cert:
            subject_match = re.search(r"Subject:.*?CN\s*=\s*([^,\n]+)", ssl_cert, re.I | re.S)
            if subject_match:
                cert_cn = subject_match.group(1).strip().lower()
                if clean_target.lower() not in cert_cn:
                    findings.append(SecurityFinding(
                        title="Certificate CN Mismatch",
                        description="Certificate common name does not match the requested host.",
                        severity=SeverityLevel.MEDIUM,
                        category="ssl",
                        evidence=f"Certificate CN '{cert_cn}' vs host '{clean_target}'.",
                        recommendation="Issue a certificate whose SAN list includes the target domain."
                    ))

        # WHOIS domain expiry
        whois_data = recon_data.get("whois", "")
        if whois_data:
            expiry = self._extract_whois_expiry(whois_data)
            if expiry:
                days_remaining = (expiry - datetime.utcnow()).days
                if days_remaining < 0:
                    severity = SeverityLevel.HIGH
                    description = "Domain appears past its registry expiration date."
                elif days_remaining < 60:
                    severity = SeverityLevel.MEDIUM
                    description = f"Domain registration expires in {days_remaining} days."
                else:
                    severity = None
                    description = ""

                if severity:
                    findings.append(SecurityFinding(
                        title="Domain Renewal Window Approaching",
                        description=description,
                        severity=severity,
                        category="dns",
                        evidence=whois_data.splitlines()[0][:200],
                        recommendation="Renew the domain registration and ensure auto-renew is configured."
                    ))

        return findings

    def _extract_header_value(self, headers_raw: str, header_name: str) -> Optional[str]:
        """Extract a header value from raw HTTP headers."""
        if not headers_raw:
            return None

        header_lower = header_name.lower()
        for line in headers_raw.splitlines():
            if line.lower().startswith(header_lower + ":"):
                return line.split(":", 1)[1].strip()
        return None

    def _extract_whois_expiry(self, whois_output: str) -> Optional[datetime]:
        """Parse WHOIS output for domain expiry date"""
        patterns = [
            r"(Expiration Date|Registry Expiry Date):\s*([0-9T:\-\.Z ]+)",
            r"paid-till:\s*([0-9\-]+)"
        ]
        for pattern in patterns:
            match = re.search(pattern, whois_output, re.IGNORECASE)
            if match:
                date_str = match.group(len(match.groups()))
                for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y.%m.%d"):
                    try:
                        return datetime.strptime(date_str.strip(), fmt)
                    except ValueError:
                        continue
        return None

    def _generate_summary(self, target: str, severity_counts: Dict[str, int]) -> str:
        """Generate executive summary of findings"""
        critical = severity_counts["critical"]
        high = severity_counts["high"]
        medium = severity_counts["medium"]
        low = severity_counts["low"]
        info = severity_counts["info"]

        total = sum(severity_counts.values())

        summary = f"""Security Assessment Summary for {target}

Total Findings: {total}
  - Critical: {critical}
  - High: {high}
  - Medium: {medium}
  - Low: {low}
  - Informational: {info}

Priority Actions:
"""

        if critical > 0:
            summary += f"  🔴 URGENT: Address {critical} critical finding(s) immediately\n"
        if high > 0:
            summary += f"  🟠 HIGH: Remediate {high} high-severity finding(s) soon\n"
        if medium > 0:
            summary += f"  🟡 MEDIUM: Plan fixes for {medium} medium-severity finding(s)\n"

        if critical == 0 and high == 0:
            summary += "  ✅ No critical or high-severity issues detected\n"

        return summary

    def _generate_basic_report(self, target: str, phases_completed: List[str]) -> SecurityReport:
        """Generate basic report when full reporting phase not run"""
        return self._phase_reporting(target, phases_completed)

    def print_report(self, report: SecurityReport):
        """Print report to console in human-readable format"""
        print("\n" + "="*70)
        print(f"🔒 SECURITY ASSESSMENT REPORT")
        print("="*70)
        print(f"\nTarget: {report.target}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Phases: {', '.join(report.phases_completed)}")
        print(f"\n{report.summary}")

        if report.findings:
            print("\n" + "-"*70)
            print("DETAILED FINDINGS")
            print("-"*70 + "\n")

            # Group by severity
            for severity in ["critical", "high", "medium", "low", "info"]:
                severity_findings = [f for f in report.findings if f.severity.value == severity]
                if severity_findings:
                    severity_icon = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🔵",
                        "info": "ℹ️"
                    }[severity]

                    print(f"\n{severity_icon} {severity.upper()} FINDINGS ({len(severity_findings)}):")
                    print("-" * 70)

                    for i, finding in enumerate(severity_findings, 1):
                        print(f"\n{i}. {finding.title}")
                        print(f"   Category: {finding.category}")
                        print(f"   Description: {finding.description}")
                        print(f"   Evidence: {finding.evidence}")
                        print(f"   Recommendation: {finding.recommendation}")
                        if finding.cve_ids:
                            print(f"   CVEs: {', '.join(finding.cve_ids)}")

        print("\n" + "="*70 + "\n")


# Example usage and CLI
if __name__ == "__main__":
    import sys

    # Check for flags
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("Security Agent - Autonomous Security Testing")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 security_agent.py <target>")
        print("  python3 security_agent.py --list-tools")
        print("\nOptions:")
        print("  --list-tools    Show available security tools")
        print("  -h, --help      Show this help message")
        print("\nExamples:")
        print("  python3 security_agent.py example.com")
        print("  python3 security_agent.py https://example.com")
        print("\n⚠️  AUTHORIZED TESTING ONLY")
        print("Only test systems you own or have explicit permission to test.")
        print("=" * 60)
        sys.exit(0 if sys.argv[1] in ["-h", "--help"] else 1)

    # Handle --list-tools flag
    if sys.argv[1] == "--list-tools":
        agent = SecurityAgent()
        agent.tool_registry.print_status()
        print("\n💡 To add more tools:")
        print("   1. Create JSON descriptor in security_tools/tool_descriptors/")
        print("   2. Install tool or add script to security_tools/tool_scripts/")
        print("   See security_tools/README.md for details.\n")
        sys.exit(0)

    target = sys.argv[1]

    print("\n🔒 DSMIL Security Agent - Autonomous Assessment")
    print("⚠️  Ensure you have authorization to test this target!\n")

    # Create agent
    agent = SecurityAgent()

    # Show available tools
    available_tools = len(agent.tool_registry.list_tools(only_available=True))
    total_tools = len(agent.tool_registry.list_tools())
    print(f"🔧 Tools Available: {available_tools}/{total_tools}")
    if available_tools < total_tools:
        print(f"   (Run with --list-tools to see which tools are missing)\n")
    else:
        print()

    # Run assessment
    report = agent.assess_target(target)

    # Print report
    agent.print_report(report)

    # Save report to file
    output_file = Path(f"security_report_{target.replace('://', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report.to_json(output_file)
    print(f"📄 Report saved to: {output_file}")
