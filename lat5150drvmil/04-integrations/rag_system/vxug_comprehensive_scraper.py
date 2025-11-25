#!/usr/bin/env python3
"""
VX Underground Comprehensive Archive Scraper

Downloads the complete VX Underground collection (800+ papers):
- APT Reports (500+ reports, 58 APT groups, 2010-2024)
- Malware Analysis Papers (300+ papers)
- Technique Papers (from GitHub)

Designed for one-time bulk download with respectful rate limiting.
Approved by VX Underground (SmellyVX) for bulk download.

Usage:
    # Download everything (recommended)
    python3 vxug_comprehensive_scraper.py --all

    # Download specific collections
    python3 vxug_comprehensive_scraper.py --apt-reports
    python3 vxug_comprehensive_scraper.py --malware-papers

    # Resume interrupted download
    python3 vxug_comprehensive_scraper.py --resume

    # Show progress
    python3 vxug_comprehensive_scraper.py --status
"""

import os
import re
import json
import time
import hashlib
import logging
import asyncio
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from html.parser import HTMLParser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/vxug_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
VX_BASE_DIR = Path('00-documentation/Security_Feed/VX_Underground')
VX_APT_DIR = VX_BASE_DIR / 'APT_Reports'
VX_MALWARE_DIR = VX_BASE_DIR / 'Malware_Analysis'
VX_INDEX_FILE = Path('rag_system/vxug_comprehensive_index.json')

# VX Underground URLs
VX_URLS = {
    'base': 'https://vx-underground.org',
    'apts': 'https://vx-underground.org/APTs',
    'papers': 'https://vx-underground.org/papers',
    'samples': 'https://samples.vx-underground.org'
}

# APT Report years available
APT_YEARS = list(range(2010, 2025))  # 2010-2024

# Rate limiting (respectful but not overly restrictive)
DELAY_BETWEEN_REQUESTS = 2  # 2 seconds between requests
DELAY_BETWEEN_FILES = 1  # 1 second between file downloads
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 60
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max per file

# Create directories
for dir in [VX_BASE_DIR, VX_APT_DIR, VX_MALWARE_DIR]:
    dir.mkdir(parents=True, exist_ok=True)


class LinkExtractor(HTMLParser):
    """Simple HTML parser to extract links"""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    self.links.append(value)


@dataclass
class VXPaper:
    """VX Underground paper metadata"""
    id: str
    title: str
    url: str
    filename: str
    path: str
    category: str  # apt_report, malware_analysis, technique
    subcategory: str  # APT group name, malware family, etc.
    year: Optional[int] = None
    size: int = 0
    hash: str = ''
    downloaded_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class VXUndergroundScraper:
    """Comprehensive VX Underground scraper"""

    def __init__(self):
        self.papers: Dict[str, VXPaper] = {}
        self.downloaded_hashes: Set[str] = set()
        self.failed_downloads: List[str] = []
        self.index = self._load_index()
        self.session_downloads = 0
        self.session_bytes = 0

        logger.info("=" * 80)
        logger.info("VX Underground Comprehensive Scraper")
        logger.info("=" * 80)
        logger.info("Target: 800+ papers (APT reports + malware analysis)")
        logger.info("Rate limit: 2s between requests, 1s between files")
        logger.info("Approved by: SmellyVX (VX Underground)")
        logger.info("=" * 80)

    def _load_index(self) -> Dict:
        """Load existing download index"""
        if VX_INDEX_FILE.exists():
            with open(VX_INDEX_FILE, 'r') as f:
                data = json.load(f)
                # Rebuild papers dict
                for paper_id, paper_data in data.get('papers', {}).items():
                    self.papers[paper_id] = VXPaper(**paper_data)
                    if paper_data.get('hash'):
                        self.downloaded_hashes.add(paper_data['hash'])
                return data
        return {
            'papers': {},
            'progress': {},
            'statistics': {},
            'last_update': None
        }

    def _save_index(self):
        """Save download index"""
        self.index['papers'] = {id: paper.to_dict() for id, paper in self.papers.items()}
        self.index['last_update'] = datetime.now().isoformat()
        self.index['statistics'] = {
            'total_papers': len(self.papers),
            'total_bytes': sum(p.size for p in self.papers.values()),
            'by_category': {},
            'by_year': {},
            'session_downloads': self.session_downloads,
            'session_bytes': self.session_bytes
        }

        # Calculate statistics
        for paper in self.papers.values():
            cat = paper.category
            self.index['statistics']['by_category'][cat] = \
                self.index['statistics']['by_category'].get(cat, 0) + 1

            if paper.year:
                self.index['statistics']['by_year'][str(paper.year)] = \
                    self.index['statistics']['by_year'].get(str(paper.year), 0) + 1

        with open(VX_INDEX_FILE, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.info(f"Index saved: {len(self.papers)} papers")

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _fetch_url(self, url: str) -> Optional[bytes]:
        """Fetch URL content with retries"""
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (VX-Underground Archive Bot)',
                        'Accept': '*/*'
                    }
                )

                with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
                    return response.read()

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.debug(f"Not found (404): {url}")
                    return None
                elif e.code == 429:
                    wait_time = (attempt + 1) * 10
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP {e.code}: {url}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))

        return None

    def _extract_links(self, html: bytes, base_url: str) -> List[str]:
        """Extract links from HTML"""
        try:
            html_str = html.decode('utf-8', errors='ignore')
            parser = LinkExtractor()
            parser.feed(html_str)

            # Filter and normalize links
            links = []
            for link in parser.links:
                # Skip navigation links
                if link in ['..', './', '../', '#']:
                    continue

                # Make absolute
                if link.startswith('http'):
                    links.append(link)
                elif link.startswith('/'):
                    links.append(VX_URLS['base'] + link)
                else:
                    links.append(base_url.rstrip('/') + '/' + link)

            return links
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []

    def _download_file(self, url: str, dest_path: Path, paper_id: str) -> bool:
        """Download a single file"""
        try:
            # Check if already downloaded
            if dest_path.exists():
                logger.debug(f"Already exists: {dest_path.name}")
                return True

            logger.info(f"Downloading: {dest_path.name}")

            # Download
            data = self._fetch_url(url)
            if not data:
                self.failed_downloads.append(url)
                return False

            # Check size
            if len(data) > MAX_FILE_SIZE:
                logger.warning(f"File too large ({len(data)} bytes): {dest_path.name}")
                return False

            # Check for duplicate by hash
            temp_path = dest_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(data)

            file_hash = self._get_file_hash(temp_path)

            if file_hash in self.downloaded_hashes:
                logger.info(f"Duplicate (hash match): {dest_path.name}")
                temp_path.unlink()
                return True

            # Move to final location
            temp_path.rename(dest_path)

            # Update metadata
            if paper_id in self.papers:
                self.papers[paper_id].size = len(data)
                self.papers[paper_id].hash = file_hash
                self.papers[paper_id].downloaded_at = datetime.now().isoformat()

            self.downloaded_hashes.add(file_hash)
            self.session_downloads += 1
            self.session_bytes += len(data)

            logger.info(f"✓ Downloaded: {dest_path.name} ({len(data):,} bytes)")

            # Save index periodically
            if self.session_downloads % 10 == 0:
                self._save_index()

            time.sleep(DELAY_BETWEEN_FILES)
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            self.failed_downloads.append(url)
            return False

    async def scrape_apt_reports(self):
        """Scrape all APT reports from all years"""
        logger.info("=" * 80)
        logger.info("Scraping APT Reports")
        logger.info("=" * 80)

        for year in APT_YEARS:
            year_url = f"{VX_URLS['apts']}/{year}/"
            logger.info(f"Fetching APT reports for {year}...")

            # Get directory listing
            html = self._fetch_url(year_url)
            if not html:
                logger.warning(f"Could not fetch {year} directory")
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue

            links = self._extract_links(html, year_url)

            # Filter for PDF files
            pdf_links = [l for l in links if l.lower().endswith('.pdf')]

            logger.info(f"Found {len(pdf_links)} PDFs for {year}")

            for pdf_url in pdf_links:
                filename = pdf_url.split('/')[-1]
                filename = urllib.parse.unquote(filename)

                # Create paper metadata
                paper_id = f"apt_{year}_{hashlib.md5(pdf_url.encode()).hexdigest()[:8]}"

                # Extract APT group from URL/filename
                apt_group = "Unknown"
                for part in pdf_url.split('/'):
                    if 'apt' in part.lower() or any(g.lower() in part.lower() for g in ['lazarus', 'turla', 'fin']):
                        apt_group = part
                        break

                dest_path = VX_APT_DIR / str(year) / filename
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                paper = VXPaper(
                    id=paper_id,
                    title=filename,
                    url=pdf_url,
                    filename=filename,
                    path=str(dest_path),
                    category='apt_report',
                    subcategory=apt_group,
                    year=year
                )

                self.papers[paper_id] = paper

                # Download
                self._download_file(pdf_url, dest_path, paper_id)

                time.sleep(DELAY_BETWEEN_REQUESTS)

            # Save progress after each year
            self._save_index()

        logger.info(f"APT reports complete: {self.session_downloads} files downloaded")

    async def scrape_malware_papers(self):
        """Scrape malware analysis papers"""
        logger.info("=" * 80)
        logger.info("Scraping Malware Analysis Papers")
        logger.info("=" * 80)

        papers_url = f"{VX_URLS['papers']}"

        logger.info(f"Fetching malware papers index...")
        html = self._fetch_url(papers_url)

        if not html:
            logger.error("Could not fetch malware papers index")
            return

        links = self._extract_links(html, papers_url)
        pdf_links = [l for l in links if l.lower().endswith('.pdf')]

        logger.info(f"Found {len(pdf_links)} malware analysis papers")

        for pdf_url in pdf_links:
            filename = pdf_url.split('/')[-1]
            filename = urllib.parse.unquote(filename)

            paper_id = f"malware_{hashlib.md5(pdf_url.encode()).hexdigest()[:8]}"

            dest_path = VX_MALWARE_DIR / filename

            paper = VXPaper(
                id=paper_id,
                title=filename,
                url=pdf_url,
                filename=filename,
                path=str(dest_path),
                category='malware_analysis',
                subcategory='general'
            )

            self.papers[paper_id] = paper

            self._download_file(pdf_url, dest_path, paper_id)

            time.sleep(DELAY_BETWEEN_REQUESTS)

        self._save_index()
        logger.info(f"Malware papers complete: {self.session_downloads} files downloaded")

    async def download_all(self):
        """Download everything"""
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting comprehensive VX Underground download")
        logger.info("Target: 800+ papers")
        logger.info("=" * 80)

        # APT Reports
        await self.scrape_apt_reports()

        # Malware Analysis
        await self.scrape_malware_papers()

        # Final save
        self._save_index()

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("Download Complete")
        logger.info("=" * 80)
        logger.info(f"Downloaded: {self.session_downloads} files")
        logger.info(f"Total size: {self.session_bytes / (1024*1024):.2f} MB")
        logger.info(f"Time elapsed: {elapsed / 60:.1f} minutes")
        logger.info(f"Failed: {len(self.failed_downloads)} files")
        logger.info("=" * 80)

        if self.failed_downloads:
            logger.warning(f"Failed downloads ({len(self.failed_downloads)}):")
            for url in self.failed_downloads[:10]:  # Show first 10
                logger.warning(f"  - {url}")

    def show_status(self):
        """Show download status"""
        stats = self.index.get('statistics', {})

        print("\n" + "=" * 80)
        print("VX Underground Download Status")
        print("=" * 80)
        print(f"\nTotal Papers: {stats.get('total_papers', 0)}")
        print(f"Total Size: {stats.get('total_bytes', 0) / (1024*1024):.2f} MB")
        print(f"Last Update: {self.index.get('last_update', 'Never')}")

        print("\nBy Category:")
        for cat, count in sorted(stats.get('by_category', {}).items()):
            print(f"  {cat:20s}: {count:4d} papers")

        print("\nBy Year:")
        for year, count in sorted(stats.get('by_year', {}).items()):
            print(f"  {year:10s}: {count:4d} papers")

        print()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='VX Underground Comprehensive Scraper')
    parser.add_argument('--all', action='store_true', help='Download everything')
    parser.add_argument('--apt-reports', action='store_true', help='Download APT reports only')
    parser.add_argument('--malware-papers', action='store_true', help='Download malware analysis only')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted download')
    parser.add_argument('--status', action='store_true', help='Show download status')

    args = parser.parse_args()

    scraper = VXUndergroundScraper()

    if args.status:
        scraper.show_status()
        return

    if args.all or args.resume:
        await scraper.download_all()
    elif args.apt_reports:
        await scraper.scrape_apt_reports()
    elif args.malware_papers:
        await scraper.scrape_malware_papers()
    else:
        scraper.show_status()
        print("\nRun with --all to download everything")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nScraper stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
