#!/usr/bin/env python3
"""
Threat Intelligence Automation
Automated IOC extraction, threat actor attribution, and campaign tracking

Key Capabilities:
- IOC (Indicator of Compromise) extraction
- Threat actor attribution
- Campaign tracking and correlation
- Automated TI reporting
- Integration with DIRECTEYE threat intel

Use Cases:
- Extract IOCs from incident reports
- Attribute attacks to known threat actors
- Track campaigns across multiple incidents
- Generate threat intelligence reports
- Automated threat hunting
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import Counter


class IOCType(Enum):
    """Types of Indicators of Compromise"""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    EMAIL = "email"
    FILE_HASH_MD5 = "md5"
    FILE_HASH_SHA1 = "sha1"
    FILE_HASH_SHA256 = "sha256"
    CVE = "cve"
    MUTEX = "mutex"
    REGISTRY_KEY = "registry_key"
    FILE_PATH = "file_path"
    BITCOIN_ADDRESS = "bitcoin_address"
    ETHEREUM_ADDRESS = "ethereum_address"


class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IOC:
    """
    Indicator of Compromise

    Attributes:
        value: The IOC value (IP, domain, hash, etc.)
        ioc_type: Type of IOC
        confidence: Confidence score (0.0-1.0)
        first_seen: When this IOC was first observed
        last_seen: When this IOC was last observed
        threat_level: Severity level
        context: Additional context about the IOC
        tags: Associated tags (malware family, campaign, etc.)
    """
    value: str
    ioc_type: IOCType
    confidence: float = 0.8
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    context: str = ""
    tags: Set[str] = field(default_factory=set)


@dataclass
class ThreatCampaign:
    """
    Tracked threat campaign

    Attributes:
        campaign_id: Unique campaign identifier
        name: Campaign name
        threat_actor: Attributed threat actor
        iocs: Associated IOCs
        first_seen: Campaign start date
        last_seen: Most recent activity
        targets: Targeted industries/regions
        ttps: Tactics, Techniques, and Procedures
        description: Campaign description
    """
    campaign_id: str
    name: str
    threat_actor: Optional[str] = None
    iocs: List[IOC] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    targets: List[str] = field(default_factory=list)
    ttps: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ThreatReport:
    """
    Automated threat intelligence report

    Attributes:
        report_id: Unique report identifier
        title: Report title
        summary: Executive summary
        iocs: IOCs found
        campaigns: Related campaigns
        threat_actors: Implicated threat actors
        recommendations: Recommended actions
        generated_at: Report generation timestamp
    """
    report_id: str
    title: str
    summary: str
    iocs: List[IOC]
    campaigns: List[ThreatCampaign]
    threat_actors: List[str]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class ThreatIntelligenceAutomation:
    """
    Automated threat intelligence processing

    Features:
    - IOC extraction from text
    - Threat actor attribution
    - Campaign tracking
    - Automated reporting
    - DIRECTEYE integration
    """

    def __init__(self, directeye_intel=None, event_driven_agent=None):
        """
        Initialize threat intelligence automation

        Args:
            directeye_intel: Optional DIRECTEYE Intelligence instance
            event_driven_agent: Optional event-driven agent for audit logging
        """
        self.directeye_intel = directeye_intel
        self.event_driven_agent = event_driven_agent
        self.ioc_database: Dict[str, IOC] = {}
        self.campaigns: Dict[str, ThreatCampaign] = {}
        self.threat_actors: Dict[str, Dict[str, Any]] = {}
        self.reports: List[ThreatReport] = []

        # Known threat actor patterns (simplified)
        self.threat_actor_indicators = self._load_threat_actor_patterns()

    def _load_threat_actor_patterns(self) -> Dict[str, List[str]]:
        """Load known threat actor patterns"""
        return {
            "APT28": ["fancy bear", "sofacy", "sednit", "pawn storm"],
            "APT29": ["cozy bear", "dukes", "yttrium"],
            "Lazarus Group": ["hidden cobra", "zinc", "guardians of peace"],
            "Equation Group": ["equation", "longhorn"],
            "FIN7": ["carbanak", "anunak"],
            "REvil": ["sodinokibi", "revil ransomware"],
            "Conti": ["conti ransomware", "wizard spider"],
            "LockBit": ["lockbit", "abcd ransomware"],
            "BlackCat": ["alphv", "blackcat ransomware"],
            "APT41": ["barium", "winnti", "wicked panda"]
        }

    def extract_iocs(self, text: str, context: str = "") -> List[IOC]:
        """
        Extract IOCs from text

        Args:
            text: Text to analyze
            context: Optional context about the text

        Returns:
            List of extracted IOCs
        """
        iocs = []

        # Extract different IOC types
        iocs.extend(self._extract_ip_addresses(text, context))
        iocs.extend(self._extract_domains(text, context))
        iocs.extend(self._extract_urls(text, context))
        iocs.extend(self._extract_emails(text, context))
        iocs.extend(self._extract_file_hashes(text, context))
        iocs.extend(self._extract_cves(text, context))
        iocs.extend(self._extract_crypto_addresses(text, context))

        # Store in database
        for ioc in iocs:
            self._store_ioc(ioc)

        return iocs

    def _extract_ip_addresses(self, text: str, context: str) -> List[IOC]:
        """Extract IP addresses"""
        pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(pattern, text)

        iocs = []
        for ip in ips:
            # Skip private/loopback IPs
            if ip.startswith(('127.', '192.168.', '10.', '172.')):
                continue

            ioc = IOC(
                value=ip,
                ioc_type=IOCType.IP_ADDRESS,
                context=context,
                tags={"ipaddress"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_domains(self, text: str, context: str) -> List[IOC]:
        """Extract domain names"""
        pattern = r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b'
        domains = re.findall(pattern, text, re.IGNORECASE)

        iocs = []
        for domain in set(domains):
            # Skip common legitimate domains
            if any(d in domain.lower() for d in ['google.com', 'microsoft.com', 'apple.com', 'example.com']):
                continue

            ioc = IOC(
                value=domain.lower(),
                ioc_type=IOCType.DOMAIN,
                context=context,
                tags={"domain"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_urls(self, text: str, context: str) -> List[IOC]:
        """Extract URLs"""
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(pattern, text, re.IGNORECASE)

        iocs = []
        for url in set(urls):
            ioc = IOC(
                value=url,
                ioc_type=IOCType.URL,
                context=context,
                tags={"url"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_emails(self, text: str, context: str) -> List[IOC]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)

        iocs = []
        for email in set(emails):
            ioc = IOC(
                value=email.lower(),
                ioc_type=IOCType.EMAIL,
                context=context,
                tags={"email"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_file_hashes(self, text: str, context: str) -> List[IOC]:
        """Extract file hashes (MD5, SHA1, SHA256)"""
        iocs = []

        # MD5 (32 hex chars)
        md5_pattern = r'\b[a-fA-F0-9]{32}\b'
        for md5 in re.findall(md5_pattern, text):
            ioc = IOC(
                value=md5.lower(),
                ioc_type=IOCType.FILE_HASH_MD5,
                context=context,
                tags={"hash", "md5"}
            )
            iocs.append(ioc)

        # SHA1 (40 hex chars)
        sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
        for sha1 in re.findall(sha1_pattern, text):
            ioc = IOC(
                value=sha1.lower(),
                ioc_type=IOCType.FILE_HASH_SHA1,
                context=context,
                tags={"hash", "sha1"}
            )
            iocs.append(ioc)

        # SHA256 (64 hex chars)
        sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
        for sha256 in re.findall(sha256_pattern, text):
            ioc = IOC(
                value=sha256.lower(),
                ioc_type=IOCType.FILE_HASH_SHA256,
                context=context,
                tags={"hash", "sha256"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_cves(self, text: str, context: str) -> List[IOC]:
        """Extract CVE identifiers"""
        pattern = r'CVE-\d{4}-\d{4,}'
        cves = re.findall(pattern, text, re.IGNORECASE)

        iocs = []
        for cve in set(cves):
            ioc = IOC(
                value=cve.upper(),
                ioc_type=IOCType.CVE,
                context=context,
                tags={"cve", "vulnerability"}
            )
            iocs.append(ioc)

        return iocs

    def _extract_crypto_addresses(self, text: str, context: str) -> List[IOC]:
        """Extract cryptocurrency addresses"""
        iocs = []

        # Bitcoin addresses
        btc_pattern = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'
        for btc in re.findall(btc_pattern, text):
            ioc = IOC(
                value=btc,
                ioc_type=IOCType.BITCOIN_ADDRESS,
                context=context,
                tags={"crypto", "bitcoin"}
            )
            iocs.append(ioc)

        # Ethereum addresses
        eth_pattern = r'\b0x[a-fA-F0-9]{40}\b'
        for eth in re.findall(eth_pattern, text):
            ioc = IOC(
                value=eth,
                ioc_type=IOCType.ETHEREUM_ADDRESS,
                context=context,
                tags={"crypto", "ethereum"}
            )
            iocs.append(ioc)

        return iocs

    def _store_ioc(self, ioc: IOC):
        """Store IOC in database"""
        key = f"{ioc.ioc_type.value}:{ioc.value}"

        if key in self.ioc_database:
            # Update last_seen
            existing = self.ioc_database[key]
            existing.last_seen = datetime.now()
            # Merge tags
            existing.tags.update(ioc.tags)
        else:
            # New IOC
            self.ioc_database[key] = ioc

    def attribute_threat_actor(self, text: str, iocs: List[IOC]) -> List[str]:
        """
        Attribute threat actors based on text and IOCs

        Args:
            text: Incident text/report
            iocs: Associated IOCs

        Returns:
            List of likely threat actors
        """
        text_lower = text.lower()
        attributed_actors = []

        # Check for known threat actor indicators
        for actor, indicators in self.threat_actor_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    attributed_actors.append(actor)
                    break

        # TODO: More sophisticated attribution based on IOCs, TTPs, etc.
        # For now, use simple pattern matching

        return list(set(attributed_actors))

    def track_campaign(
        self,
        name: str,
        iocs: List[IOC],
        threat_actor: Optional[str] = None,
        targets: Optional[List[str]] = None,
        description: str = ""
    ) -> ThreatCampaign:
        """
        Track a threat campaign

        Args:
            name: Campaign name
            iocs: Associated IOCs
            threat_actor: Optional attributed actor
            targets: Optional list of targets
            description: Campaign description

        Returns:
            ThreatCampaign object
        """
        campaign_id = hashlib.sha256(name.encode()).hexdigest()[:16]

        if campaign_id in self.campaigns:
            # Update existing campaign
            campaign = self.campaigns[campaign_id]
            campaign.iocs.extend(iocs)
            campaign.last_seen = datetime.now()
            if threat_actor and not campaign.threat_actor:
                campaign.threat_actor = threat_actor
        else:
            # New campaign
            campaign = ThreatCampaign(
                campaign_id=campaign_id,
                name=name,
                threat_actor=threat_actor,
                iocs=iocs,
                targets=targets or [],
                description=description
            )
            self.campaigns[campaign_id] = campaign

        return campaign

    def correlate_incidents(
        self,
        incidents: List[Dict[str, Any]]
    ) -> List[ThreatCampaign]:
        """
        Correlate multiple incidents to identify campaigns

        Args:
            incidents: List of incident dicts with 'text', 'timestamp', etc.

        Returns:
            List of identified campaigns
        """
        # Extract IOCs from all incidents
        all_iocs = []
        for incident in incidents:
            iocs = self.extract_iocs(incident.get('text', ''))
            all_iocs.extend(iocs)

        # Find common IOCs (simple correlation)
        ioc_values = [f"{ioc.ioc_type.value}:{ioc.value}" for ioc in all_iocs]
        common = [ioc for ioc, count in Counter(ioc_values).items() if count > 1]

        # Group incidents with common IOCs
        if common:
            campaign = self.track_campaign(
                name=f"Campaign_{datetime.now().strftime('%Y%m%d')}",
                iocs=all_iocs,
                description=f"Correlated {len(incidents)} incidents with {len(common)} common IOCs"
            )
            return [campaign]

        return []

    def generate_report(
        self,
        title: str,
        text: str,
        auto_attribute: bool = True
    ) -> ThreatReport:
        """
        Generate automated threat intelligence report

        Args:
            title: Report title
            text: Incident text/report content
            auto_attribute: Automatically attribute threat actors

        Returns:
            ThreatReport with IOCs, attribution, recommendations
        """
        # Extract IOCs
        iocs = self.extract_iocs(text, context=title)

        # Attribute threat actors
        threat_actors = []
        if auto_attribute:
            threat_actors = self.attribute_threat_actor(text, iocs)

        # Find related campaigns
        campaigns = [c for c in self.campaigns.values() if any(ioc in c.iocs for ioc in iocs)]

        # Generate recommendations
        recommendations = self._generate_recommendations(iocs, threat_actors)

        # Create summary
        summary = self._generate_summary(iocs, threat_actors, campaigns)

        report = ThreatReport(
            report_id=hashlib.sha256(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            title=title,
            summary=summary,
            iocs=iocs,
            campaigns=campaigns,
            threat_actors=threat_actors,
            recommendations=recommendations
        )

        self.reports.append(report)

        return report

    def _generate_summary(
        self,
        iocs: List[IOC],
        threat_actors: List[str],
        campaigns: List[ThreatCampaign]
    ) -> str:
        """Generate executive summary"""
        parts = []

        # IOC count
        ioc_counts = Counter([ioc.ioc_type.value for ioc in iocs])
        ioc_summary = ", ".join([f"{count} {typ}" for typ, count in ioc_counts.items()])
        parts.append(f"Identified {len(iocs)} IOCs: {ioc_summary}")

        # Threat actors
        if threat_actors:
            parts.append(f"Attributed to: {', '.join(threat_actors)}")

        # Campaigns
        if campaigns:
            parts.append(f"Related to {len(campaigns)} known campaigns")

        return ". ".join(parts) + "."

    def _generate_recommendations(
        self,
        iocs: List[IOC],
        threat_actors: List[str]
    ) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        # Block IOCs
        if iocs:
            recommendations.append(f"Block {len(iocs)} identified IOCs in firewall/IDS")

        # IOC-specific recommendations
        for ioc in iocs:
            if ioc.ioc_type == IOCType.IP_ADDRESS:
                recommendations.append(f"Block IP {ioc.value} at network perimeter")
            elif ioc.ioc_type == IOCType.DOMAIN:
                recommendations.append(f"Block domain {ioc.value} in DNS/web proxy")
            elif ioc.ioc_type in [IOCType.FILE_HASH_MD5, IOCType.FILE_HASH_SHA1, IOCType.FILE_HASH_SHA256]:
                recommendations.append(f"Scan for file hash {ioc.value} in endpoints")
            elif ioc.ioc_type == IOCType.CVE:
                recommendations.append(f"Patch vulnerability {ioc.value}")

        # Actor-specific recommendations
        if threat_actors:
            recommendations.append(f"Review TTPs for {', '.join(threat_actors)} and implement controls")

        # General recommendations
        recommendations.append("Conduct threat hunting for related indicators")
        recommendations.append("Review security logs for additional IOCs")
        recommendations.append("Update threat intelligence feeds")

        return recommendations[:10]  # Top 10

    async def enrich_with_directeye(self, iocs: List[IOC]) -> List[IOC]:
        """
        Enrich IOCs with DIRECTEYE threat intelligence

        Args:
            iocs: List of IOCs to enrich

        Returns:
            Enriched IOCs with additional context
        """
        if not self.directeye_intel:
            return iocs

        for ioc in iocs:
            # Query DIRECTEYE based on IOC type
            if ioc.ioc_type == IOCType.IP_ADDRESS:
                # IP threat intel
                result = await self.directeye_intel.threat_intel_query(ioc.value, query_type="ip")
                if result:
                    ioc.threat_level = self._parse_threat_level(result.get('threat_score', 0))
                    ioc.tags.update(result.get('tags', []))
                    ioc.context += f"; DIRECTEYE: {result.get('description', '')}"

            elif ioc.ioc_type == IOCType.DOMAIN:
                # Domain threat intel
                result = await self.directeye_intel.threat_intel_query(ioc.value, query_type="domain")
                if result:
                    ioc.threat_level = self._parse_threat_level(result.get('threat_score', 0))
                    ioc.tags.update(result.get('tags', []))

        return iocs

    def _parse_threat_level(self, score: float) -> ThreatLevel:
        """Parse threat score to threat level"""
        if score >= 0.8:
            return ThreatLevel.CRITICAL
        elif score >= 0.6:
            return ThreatLevel.HIGH
        elif score >= 0.4:
            return ThreatLevel.MEDIUM
        elif score >= 0.2:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO

    def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics"""
        ioc_counts = Counter([ioc.ioc_type.value for ioc in self.ioc_database.values()])

        return {
            "total_iocs": len(self.ioc_database),
            "ioc_types": dict(ioc_counts),
            "campaigns_tracked": len(self.campaigns),
            "threat_actors_identified": len([c.threat_actor for c in self.campaigns.values() if c.threat_actor]),
            "reports_generated": len(self.reports)
        }


def main():
    """Demo usage"""
    print("=== Threat Intelligence Automation Demo ===\n")

    ti = ThreatIntelligenceAutomation()

    # Test text with IOCs
    incident_text = """
    Detected suspicious activity from IP 192.0.2.1 connecting to evil-domain.com.
    File hash: 5d41402abc4b2a76b9719d911017c592
    Attacker used CVE-2021-44228 (Log4Shell) for initial access.
    Ransomware payment demanded to bitcoin address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.
    Indicators suggest Conti ransomware campaign.
    """

    # Extract IOCs
    print("1. Extracting IOCs...")
    iocs = ti.extract_iocs(incident_text, context="Ransomware incident")
    print(f"   Found {len(iocs)} IOCs:")
    for ioc in iocs:
        print(f"     - {ioc.ioc_type.value}: {ioc.value}")

    # Attribute threat actor
    print("\n2. Attributing threat actor...")
    actors = ti.attribute_threat_actor(incident_text, iocs)
    print(f"   Attributed to: {actors}")

    # Generate report
    print("\n3. Generating threat intelligence report...")
    report = ti.generate_report(
        title="Conti Ransomware Incident",
        text=incident_text,
        auto_attribute=True
    )
    print(f"   Report ID: {report.report_id}")
    print(f"   Summary: {report.summary}")
    print(f"   Recommendations ({len(report.recommendations)}):")
    for i, rec in enumerate(report.recommendations[:5], 1):
        print(f"     {i}. {rec}")

    # Statistics
    print("\n4. Statistics:")
    stats = ti.get_statistics()
    print(f"   Total IOCs: {stats['total_iocs']}")
    print(f"   IOC types: {stats['ioc_types']}")
    print(f"   Reports: {stats['reports_generated']}")


if __name__ == "__main__":
    main()
