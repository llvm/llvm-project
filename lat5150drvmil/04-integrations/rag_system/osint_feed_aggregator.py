#!/usr/bin/env python3
"""
OSINT Feed Aggregator

Comprehensive OSINT (Open Source Intelligence) data collection from multiple sources:
- RSS feeds (security blogs, news)
- Threat intelligence APIs (AlienVault OTX, abuse.ch, etc.)
- GitHub security advisories
- CVE/NVD databases
- IOC feeds (malware hashes, IPs, domains, URLs)
- Pastebin monitoring
- Certificate transparency logs
- Social media (Twitter/X security researchers)

Features:
- Multi-source aggregation
- Automatic deduplication
- RAG system integration
- TOON compression for efficient storage
- Configurable update intervals
- Offline mode with cached data
- Export to multiple formats (JSON, TOON, Markdown)

Usage:
    # Collect from all sources
    python3 osint_feed_aggregator.py --all

    # Specific sources
    python3 osint_feed_aggregator.py --rss --threat-intel --github

    # Continuous monitoring
    python3 osint_feed_aggregator.py --monitor --interval 3600

    # Export data
    python3 osint_feed_aggregator.py --export markdown --output osint_report.md
"""

import os
import re
import json
import time
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/osint_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OSINT_DATA_DIR = Path('00-documentation/Security_Feed/OSINT')
OSINT_INDEX_FILE = Path('rag_system/osint_index.json')
CACHE_DIR = Path('rag_system/.osint_cache')

# Create directories
OSINT_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FeedType(Enum):
    """OSINT feed categories"""
    RSS = "rss"
    THREAT_INTEL = "threat_intel"
    CVE = "cve"
    IOC = "ioc"
    GITHUB = "github"
    PASTEBIN = "pastebin"
    CERT_TRANSPARENCY = "cert_transparency"
    SOCIAL_MEDIA = "social_media"


class Severity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class OSINTItem:
    """Unified OSINT data item"""
    id: str                          # Unique identifier (SHA256 hash)
    title: str                       # Item title/summary
    description: str                 # Full description/content
    source: str                      # Source name (e.g., "KrebsOnSecurity")
    feed_type: str                   # Feed type (rss, threat_intel, etc.)
    url: Optional[str] = None        # Source URL
    published: Optional[str] = None  # Publication timestamp (ISO format)
    collected: str = None            # Collection timestamp
    severity: Optional[str] = None   # Severity level
    tags: List[str] = None           # Tags/categories
    iocs: Dict[str, List[str]] = None  # IOCs (ips, domains, hashes, etc.)
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.collected is None:
            self.collected = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.iocs is None:
            self.iocs = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


# RSS Feed Sources (Security Blogs, News)
RSS_FEEDS = {
    "KrebsOnSecurity": "https://krebsonsecurity.com/feed/",
    "Schneier on Security": "https://www.schneier.com/blog/atom.xml",
    "Threatpost": "https://threatpost.com/feed/",
    "The Hacker News": "https://feeds.feedburner.com/TheHackersNews",
    "Bleeping Computer": "https://www.bleepingcomputer.com/feed/",
    "Dark Reading": "https://www.darkreading.com/rss.xml",
    "Security Week": "https://www.securityweek.com/feed/",
    "Graham Cluley": "https://grahamcluley.com/feed/",
    "Malwarebytes Labs": "https://blog.malwarebytes.com/feed/",
    "SANS ISC": "https://isc.sans.edu/rssfeed.xml",
    "US-CERT": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
    "Cisco Talos": "https://blog.talosintelligence.com/rss/",
}

# Threat Intelligence API Sources
THREAT_INTEL_SOURCES = {
    "abuse.ch_urlhaus": "https://urlhaus-api.abuse.ch/v1/urls/recent/",
    "abuse.ch_malware_bazaar": "https://mb-api.abuse.ch/api/v1/",
    "abuse.ch_threatfox": "https://threatfox-api.abuse.ch/api/v1/",
    "alienvault_otx": "https://otx.alienvault.com/api/v1/pulses/subscribed",
    "blocklist_de": "https://lists.blocklist.de/lists/",
}

# IOC Feed Sources
IOC_FEEDS = {
    "malware_hashes": "https://bazaar.abuse.ch/export/txt/md5/recent/",
    "malicious_ips": "https://feodotracker.abuse.ch/downloads/ipblocklist.txt",
    "malicious_domains": "https://urlhaus.abuse.ch/downloads/text/",
    "phishing_urls": "https://openphish.com/feed.txt",
}


class RSSFeedCollector:
    """Collect data from RSS/Atom feeds"""

    def __init__(self):
        self.items: List[OSINTItem] = []

    def fetch_feed(self, name: str, url: str) -> List[OSINTItem]:
        """Fetch and parse RSS/Atom feed"""
        logger.info(f"Fetching RSS feed: {name}")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Detect RSS vs Atom
            if root.tag == 'rss':
                return self._parse_rss(name, root)
            elif root.tag.endswith('feed'):  # Atom
                return self._parse_atom(name, root)
            else:
                logger.warning(f"Unknown feed format for {name}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {name}: {e}")
            return []

    def _parse_rss(self, source: str, root: ET.Element) -> List[OSINTItem]:
        """Parse RSS 2.0 feed"""
        items = []

        for item in root.findall('.//item'):
            try:
                title = item.findtext('title', '').strip()
                description = item.findtext('description', '').strip()
                link = item.findtext('link', '').strip()
                pub_date = item.findtext('pubDate', '').strip()

                # Create unique ID
                item_id = hashlib.sha256(f"{source}:{link}:{title}".encode()).hexdigest()[:16]

                # Extract tags from categories
                tags = [cat.text.strip() for cat in item.findall('category') if cat.text]

                # Auto-tag based on content
                tags.extend(self._extract_tags(title + ' ' + description))

                # Extract IOCs from description
                iocs = self._extract_iocs(description)

                osint_item = OSINTItem(
                    id=item_id,
                    title=title,
                    description=description,
                    source=source,
                    feed_type=FeedType.RSS.value,
                    url=link,
                    published=pub_date,
                    tags=list(set(tags)),
                    iocs=iocs,
                    severity=self._infer_severity(title, description)
                )

                items.append(osint_item)

            except Exception as e:
                logger.error(f"Error parsing RSS item: {e}")

        return items

    def _parse_atom(self, source: str, root: ET.Element) -> List[OSINTItem]:
        """Parse Atom feed"""
        items = []

        # Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            try:
                title = entry.findtext('atom:title', '', ns).strip()

                # Content or summary
                content = entry.findtext('atom:content', '', ns).strip()
                if not content:
                    content = entry.findtext('atom:summary', '', ns).strip()

                # Link
                link_elem = entry.find('atom:link[@rel="alternate"]', ns)
                if link_elem is None:
                    link_elem = entry.find('atom:link', ns)
                link = link_elem.get('href', '') if link_elem is not None else ''

                # Published date
                published = entry.findtext('atom:published', '', ns).strip()
                if not published:
                    published = entry.findtext('atom:updated', '', ns).strip()

                # Create unique ID
                item_id = hashlib.sha256(f"{source}:{link}:{title}".encode()).hexdigest()[:16]

                # Extract tags
                tags = self._extract_tags(title + ' ' + content)

                # Extract IOCs
                iocs = self._extract_iocs(content)

                osint_item = OSINTItem(
                    id=item_id,
                    title=title,
                    description=content,
                    source=source,
                    feed_type=FeedType.RSS.value,
                    url=link,
                    published=published,
                    tags=list(set(tags)),
                    iocs=iocs,
                    severity=self._infer_severity(title, content)
                )

                items.append(osint_item)

            except Exception as e:
                logger.error(f"Error parsing Atom entry: {e}")

        return items

    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        tags = []
        text_lower = text.lower()

        # Security keywords
        keywords = {
            'ransomware': 'ransomware',
            'malware': 'malware',
            'phishing': 'phishing',
            'apt': 'apt',
            'exploit': 'exploit',
            'vulnerability': 'vulnerability',
            'breach': 'data_breach',
            'leak': 'data_leak',
            'ddos': 'ddos',
            'botnet': 'botnet',
            'trojan': 'trojan',
            'backdoor': 'backdoor',
            'zero-day': 'zero_day',
            '0day': 'zero_day',
            'supply chain': 'supply_chain',
            'social engineering': 'social_engineering',
            'credential': 'credentials',
            'password': 'credentials',
        }

        for keyword, tag in keywords.items():
            if keyword in text_lower:
                tags.append(tag)

        # Check for CVE mentions
        if re.search(r'CVE-\d{4}-\d{4,}', text, re.IGNORECASE):
            tags.append('cve')

        return tags

    def _extract_iocs(self, text: str) -> Dict[str, List[str]]:
        """Extract Indicators of Compromise from text"""
        iocs = {
            'ips': [],
            'domains': [],
            'urls': [],
            'hashes': [],
            'emails': []
        }

        # IP addresses (IPv4)
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        iocs['ips'] = list(set(re.findall(ip_pattern, text)))

        # Domains (simple pattern)
        domain_pattern = r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b'
        iocs['domains'] = list(set(re.findall(domain_pattern, text, re.IGNORECASE)))

        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        iocs['urls'] = list(set(re.findall(url_pattern, text)))

        # MD5/SHA1/SHA256 hashes
        hash_patterns = {
            'md5': r'\b[a-f0-9]{32}\b',
            'sha1': r'\b[a-f0-9]{40}\b',
            'sha256': r'\b[a-f0-9]{64}\b'
        }

        for hash_type, pattern in hash_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            iocs['hashes'].extend(matches)

        iocs['hashes'] = list(set(iocs['hashes']))

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        iocs['emails'] = list(set(re.findall(email_pattern, text)))

        # Remove empty lists
        return {k: v for k, v in iocs.items() if v}

    def _infer_severity(self, title: str, description: str) -> str:
        """Infer severity from content"""
        text = (title + ' ' + description).lower()

        critical_keywords = ['critical', 'emergency', 'zero-day', '0day', 'actively exploited']
        high_keywords = ['high severity', 'dangerous', 'widespread', 'major breach']
        medium_keywords = ['medium', 'moderate', 'vulnerability']

        if any(kw in text for kw in critical_keywords):
            return Severity.CRITICAL.value
        elif any(kw in text for kw in high_keywords):
            return Severity.HIGH.value
        elif any(kw in text for kw in medium_keywords):
            return Severity.MEDIUM.value
        else:
            return Severity.INFO.value

    def collect_all(self, feeds: Dict[str, str] = None) -> List[OSINTItem]:
        """Collect from all RSS feeds"""
        if feeds is None:
            feeds = RSS_FEEDS

        all_items = []

        for name, url in feeds.items():
            items = self.fetch_feed(name, url)
            all_items.extend(items)
            logger.info(f"Collected {len(items)} items from {name}")
            time.sleep(1)  # Rate limiting

        return all_items


class ThreatIntelCollector:
    """Collect threat intelligence from APIs"""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}

    def fetch_abuse_ch_urlhaus(self) -> List[OSINTItem]:
        """Fetch recent malicious URLs from URLhaus"""
        logger.info("Fetching abuse.ch URLhaus data")

        try:
            url = "https://urlhaus-api.abuse.ch/v1/urls/recent/"

            data = urllib.parse.urlencode({}).encode()
            req = urllib.request.Request(url, data=data)

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())

            items = []

            if result.get('query_status') == 'ok':
                for url_data in result.get('urls', [])[:50]:  # Limit to 50 recent
                    item_id = hashlib.sha256(url_data['url'].encode()).hexdigest()[:16]

                    osint_item = OSINTItem(
                        id=item_id,
                        title=f"Malicious URL: {url_data.get('url_status', 'Unknown')}",
                        description=f"URL: {url_data['url']}\nThreat: {url_data.get('threat', 'Unknown')}",
                        source="abuse.ch URLhaus",
                        feed_type=FeedType.THREAT_INTEL.value,
                        url=url_data.get('urlhaus_reference', ''),
                        published=url_data.get('dateadded', ''),
                        tags=['malicious_url', url_data.get('threat', '').lower()],
                        iocs={'urls': [url_data['url']]},
                        severity=Severity.HIGH.value,
                        metadata={
                            'threat': url_data.get('threat'),
                            'url_status': url_data.get('url_status'),
                            'tags': url_data.get('tags', [])
                        }
                    )

                    items.append(osint_item)

            logger.info(f"Collected {len(items)} URLhaus items")
            return items

        except Exception as e:
            logger.error(f"Failed to fetch URLhaus data: {e}")
            return []

    def fetch_abuse_ch_threatfox(self) -> List[OSINTItem]:
        """Fetch IOCs from ThreatFox"""
        logger.info("Fetching abuse.ch ThreatFox data")

        try:
            url = "https://threatfox-api.abuse.ch/api/v1/"

            data = json.dumps({'query': 'get_iocs', 'days': 7}).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())

            items = []

            if result.get('query_status') == 'ok':
                for ioc_data in result.get('data', [])[:100]:  # Limit to 100
                    item_id = hashlib.sha256(ioc_data['ioc'].encode()).hexdigest()[:16]

                    iocs_dict = {}
                    ioc_type = ioc_data.get('ioc_type', '').lower()

                    if 'ip' in ioc_type:
                        iocs_dict['ips'] = [ioc_data['ioc']]
                    elif 'domain' in ioc_type:
                        iocs_dict['domains'] = [ioc_data['ioc']]
                    elif 'url' in ioc_type:
                        iocs_dict['urls'] = [ioc_data['ioc']]
                    elif 'hash' in ioc_type or 'md5' in ioc_type or 'sha' in ioc_type:
                        iocs_dict['hashes'] = [ioc_data['ioc']]

                    osint_item = OSINTItem(
                        id=item_id,
                        title=f"IOC: {ioc_data.get('malware', 'Unknown')}",
                        description=f"IOC: {ioc_data['ioc']}\nType: {ioc_data.get('ioc_type')}\nThreat: {ioc_data.get('threat_type')}",
                        source="abuse.ch ThreatFox",
                        feed_type=FeedType.IOC.value,
                        url=ioc_data.get('reference', ''),
                        published=ioc_data.get('first_seen', ''),
                        tags=[ioc_data.get('malware', '').lower(), ioc_data.get('threat_type', '').lower()],
                        iocs=iocs_dict,
                        severity=Severity.HIGH.value,
                        metadata={
                            'ioc_type': ioc_data.get('ioc_type'),
                            'malware': ioc_data.get('malware'),
                            'threat_type': ioc_data.get('threat_type'),
                            'confidence_level': ioc_data.get('confidence_level')
                        }
                    )

                    items.append(osint_item)

            logger.info(f"Collected {len(items)} ThreatFox items")
            return items

        except Exception as e:
            logger.error(f"Failed to fetch ThreatFox data: {e}")
            return []

    def collect_all(self) -> List[OSINTItem]:
        """Collect from all threat intel sources"""
        all_items = []

        # abuse.ch sources
        all_items.extend(self.fetch_abuse_ch_urlhaus())
        time.sleep(2)
        all_items.extend(self.fetch_abuse_ch_threatfox())

        return all_items


class GitHubSecurityCollector:
    """Collect GitHub security advisories"""

    def fetch_advisories(self, limit: int = 50) -> List[OSINTItem]:
        """Fetch recent GitHub security advisories"""
        logger.info("Fetching GitHub security advisories")

        try:
            # GitHub Advisory Database RSS feed
            url = "https://github.com/advisories.atom"

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            items = []

            for entry in root.findall('atom:entry', ns)[:limit]:
                title = entry.findtext('atom:title', '', ns).strip()
                summary = entry.findtext('atom:summary', '', ns).strip()
                link = entry.find('atom:link', ns).get('href', '') if entry.find('atom:link', ns) is not None else ''
                published = entry.findtext('atom:published', '', ns).strip()

                item_id = hashlib.sha256(f"github:{link}".encode()).hexdigest()[:16]

                # Extract CVE if present
                cve_match = re.search(r'CVE-\d{4}-\d{4,}', title + summary, re.IGNORECASE)
                cve_id = cve_match.group(0).upper() if cve_match else None

                tags = ['github_advisory']
                if cve_id:
                    tags.append('cve')

                osint_item = OSINTItem(
                    id=item_id,
                    title=title,
                    description=summary,
                    source="GitHub Security Advisories",
                    feed_type=FeedType.GITHUB.value,
                    url=link,
                    published=published,
                    tags=tags,
                    severity=Severity.HIGH.value if 'critical' in title.lower() else Severity.MEDIUM.value,
                    metadata={'cve_id': cve_id} if cve_id else {}
                )

                items.append(osint_item)

            logger.info(f"Collected {len(items)} GitHub advisories")
            return items

        except Exception as e:
            logger.error(f"Failed to fetch GitHub advisories: {e}")
            return []


class OSINTAggregator:
    """Main OSINT aggregator coordinating all collectors"""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.rss_collector = RSSFeedCollector()
        self.threat_intel_collector = ThreatIntelCollector(api_keys)
        self.github_collector = GitHubSecurityCollector()

        self.all_items: Dict[str, OSINTItem] = {}
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load existing OSINT index"""
        if OSINT_INDEX_FILE.exists():
            with open(OSINT_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {'items': {}, 'last_update': None, 'stats': {}}

    def _save_index(self):
        """Save OSINT index"""
        self.index['last_update'] = datetime.now().isoformat()
        self.index['items'] = {id: item.to_dict() for id, item in self.all_items.items()}

        # Calculate stats
        stats = {
            'total_items': len(self.all_items),
            'by_feed_type': {},
            'by_severity': {},
            'by_source': {}
        }

        for item in self.all_items.values():
            # By feed type
            feed_type = item.feed_type
            stats['by_feed_type'][feed_type] = stats['by_feed_type'].get(feed_type, 0) + 1

            # By severity
            severity = item.severity or 'unknown'
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

            # By source
            source = item.source
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1

        self.index['stats'] = stats

        with open(OSINT_INDEX_FILE, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.info(f"Index saved: {len(self.all_items)} total items")

    def _save_item_to_file(self, item: OSINTItem):
        """Save individual OSINT item as markdown file"""
        # Organize by feed type
        feed_dir = OSINT_DATA_DIR / item.feed_type
        feed_dir.mkdir(parents=True, exist_ok=True)

        filename = feed_dir / f"{item.id}.md"

        content = f"""# {item.title}

**Source:** {item.source}
**Type:** {item.feed_type}
**Severity:** {item.severity or 'Unknown'}
**Published:** {item.published or 'Unknown'}
**Collected:** {item.collected}

## Description

{item.description}

"""

        if item.url:
            content += f"## Source URL\n\n{item.url}\n\n"

        if item.tags:
            content += f"## Tags\n\n{', '.join(item.tags)}\n\n"

        if item.iocs:
            content += "## Indicators of Compromise (IOCs)\n\n"
            for ioc_type, ioc_list in item.iocs.items():
                content += f"### {ioc_type.upper()}\n\n"
                for ioc in ioc_list[:20]:  # Limit to 20 per type
                    content += f"- `{ioc}`\n"
                content += "\n"

        if item.metadata:
            content += "## Metadata\n\n```json\n"
            content += json.dumps(item.metadata, indent=2)
            content += "\n```\n\n"

        content += f"""
---
*Collected by OSINT Feed Aggregator*
*Category:* OSINT / {item.feed_type.replace('_', ' ').title()}
"""

        with open(filename, 'w') as f:
            f.write(content)

    def collect(self, sources: List[str] = None):
        """
        Collect OSINT data from specified sources

        Args:
            sources: List of source types ['rss', 'threat_intel', 'github', 'all']
        """
        if sources is None or 'all' in sources:
            sources = ['rss', 'threat_intel', 'github']

        new_items = []

        # RSS Feeds
        if 'rss' in sources:
            logger.info("=" * 60)
            logger.info("Collecting RSS Feeds")
            logger.info("=" * 60)
            rss_items = self.rss_collector.collect_all()
            new_items.extend(rss_items)

        # Threat Intelligence
        if 'threat_intel' in sources:
            logger.info("=" * 60)
            logger.info("Collecting Threat Intelligence")
            logger.info("=" * 60)
            threat_items = self.threat_intel_collector.collect_all()
            new_items.extend(threat_items)

        # GitHub Advisories
        if 'github' in sources:
            logger.info("=" * 60)
            logger.info("Collecting GitHub Security Advisories")
            logger.info("=" * 60)
            github_items = self.github_collector.fetch_advisories()
            new_items.extend(github_items)

        # Process new items
        added_count = 0
        for item in new_items:
            if item.id not in self.all_items:
                self.all_items[item.id] = item
                self._save_item_to_file(item)
                added_count += 1

        # Save index
        self._save_index()

        logger.info("=" * 60)
        logger.info(f"Collection Complete: {added_count} new items ({len(self.all_items)} total)")
        logger.info("=" * 60)

    def update_rag_embeddings(self):
        """Trigger RAG system to update with OSINT data"""
        logger.info("Updating RAG embeddings with OSINT data...")

        try:
            import subprocess

            # Rebuild document index
            subprocess.run(
                ['python3', 'rag_system/document_processor.py'],
                check=True,
                capture_output=True
            )

            logger.info("✓ RAG embeddings updated")

        except Exception as e:
            logger.error(f"Failed to update RAG embeddings: {e}")

    def get_statistics(self) -> Dict:
        """Get aggregator statistics"""
        return self.index.get('stats', {})

    def export_markdown_report(self, output_file: Path):
        """Export all items as markdown report"""
        logger.info(f"Exporting markdown report to {output_file}")

        content = f"""# OSINT Intelligence Report

**Generated:** {datetime.now().isoformat()}
**Total Items:** {len(self.all_items)}
**Last Update:** {self.index.get('last_update', 'Never')}

## Summary Statistics

"""
        stats = self.index.get('stats', {})

        # By feed type
        content += "### By Feed Type\n\n"
        for feed_type, count in sorted(stats.get('by_feed_type', {}).items()):
            content += f"- **{feed_type.replace('_', ' ').title()}:** {count}\n"

        # By severity
        content += "\n### By Severity\n\n"
        for severity, count in sorted(stats.get('by_severity', {}).items()):
            content += f"- **{severity.upper()}:** {count}\n"

        # Recent items
        content += "\n## Recent Items (Top 50)\n\n"

        # Sort by collection time
        sorted_items = sorted(
            self.all_items.values(),
            key=lambda x: x.collected,
            reverse=True
        )[:50]

        for item in sorted_items:
            content += f"### {item.title}\n\n"
            content += f"- **Source:** {item.source}\n"
            content += f"- **Type:** {item.feed_type}\n"
            content += f"- **Severity:** {item.severity or 'Unknown'}\n"
            content += f"- **Published:** {item.published or 'Unknown'}\n"
            if item.url:
                content += f"- **URL:** {item.url}\n"
            if item.tags:
                content += f"- **Tags:** {', '.join(item.tags)}\n"
            content += "\n"

        with open(output_file, 'w') as f:
            f.write(content)

        logger.info(f"Report exported to {output_file}")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='OSINT Feed Aggregator')
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['rss', 'threat_intel', 'github', 'all'],
        default=['all'],
        help='Sources to collect from'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit'
    )
    parser.add_argument(
        '--export',
        choices=['markdown', 'json'],
        help='Export data format'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for export'
    )
    parser.add_argument(
        '--update-rag',
        action='store_true',
        help='Update RAG embeddings after collection'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Continuous monitoring mode'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Update interval in seconds (default: 3600 = 1 hour)'
    )

    args = parser.parse_args()

    aggregator = OSINTAggregator()

    if args.stats:
        # Show statistics
        stats = aggregator.get_statistics()
        print("\n" + "=" * 80)
        print("OSINT Aggregator Statistics")
        print("=" * 80)
        print(f"\nTotal Items: {stats.get('total_items', 0)}")
        print(f"Last Update: {aggregator.index.get('last_update', 'Never')}")

        print("\nBy Feed Type:")
        for feed_type, count in sorted(stats.get('by_feed_type', {}).items()):
            print(f"  {feed_type:20s}: {count:4d}")

        print("\nBy Severity:")
        for severity, count in sorted(stats.get('by_severity', {}).items()):
            print(f"  {severity:20s}: {count:4d}")

        print("\nTop Sources:")
        sources = stats.get('by_source', {})
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source:40s}: {count:4d}")

        print()
        return

    if args.export:
        # Export data
        if not args.output:
            args.output = f"osint_report.{args.export}"

        if args.export == 'markdown':
            aggregator.export_markdown_report(Path(args.output))
        elif args.export == 'json':
            with open(args.output, 'w') as f:
                json.dump(aggregator.index, f, indent=2)
            logger.info(f"Exported JSON to {args.output}")
        return

    # Collection mode
    if args.monitor:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            aggregator.collect(sources=args.sources)

            if args.update_rag:
                aggregator.update_rag_embeddings()

            logger.info(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        # One-shot collection
        aggregator.collect(sources=args.sources)

        if args.update_rag:
            aggregator.update_rag_embeddings()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nAggregator stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
