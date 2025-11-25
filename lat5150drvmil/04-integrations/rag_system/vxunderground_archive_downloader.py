#!/usr/bin/env python3
"""
VX Underground Paper Archive Downloader
Downloads and indexes papers from VX Underground's research archive

VX Underground is one of the largest collections of malware source code,
samples, and research papers in the world.

Features:
- Download papers from VX Underground GitHub repository
- Organize by category (malware analysis, APT reports, techniques)
- Automatic RAG database integration
- Deduplication and indexing
- Progress tracking and resume capability
"""

import os
import re
import json
import asyncio
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
import urllib.request
import urllib.error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/vxunderground_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# VX Underground GitHub repository
VX_GITHUB_BASE = "https://raw.githubusercontent.com/vxunderground/MalwareSourceCode/main"
VX_PAPERS_BASE = "https://raw.githubusercontent.com/vxunderground/VXUG-Papers/main"

# Alternative: Direct website access (if available)
VX_WEBSITE_BASE = "https://www.vx-underground.org/papers"

# Configuration
VX_DATA_DIR = Path('00-documentation/Security_Feed/VX_Underground')
VX_INDEX_FILE = Path('rag_system/vxunderground_index.json')

# Category structure (based on VX Underground organization)
VX_CATEGORIES = {
    'APT': 'APTs',
    'Rootkits': 'Rootkits and Bootkits',
    'Ransomware': 'Ransomware',
    'Techniques': 'Techniques',
    'Malware_Analysis': 'Malware Analysis Reports',
    'Exploit_Dev': 'Exploit Development',
    'Reverse_Engineering': 'Reverse Engineering',
    'DFIR': 'Digital Forensics and Incident Response',
    'General': 'General Security Research'
}

# File settings
ALLOWED_EXTENSIONS = {'.pdf', '.md', '.txt', '.doc', '.docx'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Download settings
MAX_CONCURRENT_DOWNLOADS = 5
DOWNLOAD_TIMEOUT = 60  # seconds
RETRY_ATTEMPTS = 3


class VXUndergroundDownloader:
    """Download and index VX Underground research papers"""

    def __init__(self):
        """Initialize VX Underground downloader"""
        self.vx_index = self._load_vx_index()
        self.downloaded_hashes: Set[str] = set()
        self.new_papers = []

        # Create category directories
        VX_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for category in VX_CATEGORIES.keys():
            (VX_DATA_DIR / category).mkdir(parents=True, exist_ok=True)

        # Load downloaded files index
        self._load_downloaded_hashes()

        logger.info("VX Underground Downloader initialized")

    def _load_vx_index(self) -> Dict:
        """Load existing VX Underground paper index"""
        if VX_INDEX_FILE.exists():
            with open(VX_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {
            'papers': {},
            'categories': {},
            'total_papers': 0,
            'total_size_bytes': 0,
            'last_update': None,
            'download_progress': {}
        }

    def _save_vx_index(self):
        """Save VX Underground paper index"""
        self.vx_index['last_update'] = datetime.now().isoformat()
        self.vx_index['total_papers'] = len(self.vx_index['papers'])
        with open(VX_INDEX_FILE, 'w') as f:
            json.dump(self.vx_index, f, indent=2)

    def _load_downloaded_hashes(self):
        """Load set of already downloaded file hashes"""
        papers = self.vx_index.get('papers', {})
        self.downloaded_hashes = {paper['hash'] for paper in papers.values() if 'hash' in paper}

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem storage"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 200:
            name = name[:200]
        return name + ext

    async def download_file(self, url: str, dest_path: Path, retry: int = 0) -> Optional[Dict]:
        """
        Download a file from URL

        Args:
            url: URL to download from
            dest_path: Destination file path
            retry: Current retry attempt

        Returns:
            Dict with file metadata if successful
        """
        try:
            # Check if file already exists
            if dest_path.exists():
                logger.debug(f"File already exists: {dest_path.name}")
                return None

            logger.info(f"Downloading: {dest_path.name}")

            # Download file with timeout
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
                data = response.read()

            # Check size
            if len(data) > MAX_FILE_SIZE:
                logger.warning(f"File too large ({len(data)} bytes): {dest_path.name}")
                return None

            # Write to file
            with open(dest_path, 'wb') as f:
                f.write(data)

            # Calculate hash
            file_hash = self._get_file_hash(dest_path)

            # Check for duplicates
            if file_hash in self.downloaded_hashes:
                logger.info(f"Duplicate file (hash match): {dest_path.name}")
                dest_path.unlink()  # Delete duplicate
                return None

            self.downloaded_hashes.add(file_hash)

            file_metadata = {
                'filename': dest_path.name,
                'path': str(dest_path),
                'url': url,
                'size': len(data),
                'hash': file_hash,
                'downloaded_at': datetime.now().isoformat()
            }

            logger.info(f"✓ Downloaded: {dest_path.name} ({len(data)} bytes)")

            return file_metadata

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"File not found (404): {url}")
                return None
            elif retry < RETRY_ATTEMPTS:
                logger.warning(f"HTTP error {e.code}, retrying ({retry + 1}/{RETRY_ATTEMPTS})...")
                await asyncio.sleep(2 ** retry)  # Exponential backoff
                return await self.download_file(url, dest_path, retry + 1)
            else:
                logger.error(f"Failed to download after {RETRY_ATTEMPTS} attempts: {url}")
                return None

        except Exception as e:
            if retry < RETRY_ATTEMPTS:
                logger.warning(f"Error: {e}, retrying ({retry + 1}/{RETRY_ATTEMPTS})...")
                await asyncio.sleep(2 ** retry)
                return await self.download_file(url, dest_path, retry + 1)
            else:
                logger.error(f"Failed to download {dest_path.name}: {e}")
                return None

    def _create_paper_markdown(self, paper_data: Dict, paper_id: str):
        """Create markdown index file for paper"""
        category = paper_data.get('category', 'General')
        index_path = VX_DATA_DIR / category / f"{paper_id}_index.md"

        content = f"""# {paper_data.get('title', paper_id)}

**Category:** {VX_CATEGORIES.get(category, category)}
**Source:** VX Underground
**Downloaded:** {paper_data['downloaded_at']}

## File Information

- **Filename:** {paper_data['filename']}
- **Size:** {paper_data['size']} bytes
- **Hash (SHA256):** {paper_data['hash']}
- **Path:** `{paper_data['path']}`

## Description

{paper_data.get('description', 'Security research paper from VX Underground archive.')}

"""

        if paper_data.get('url'):
            content += f"## Source URL\n\n{paper_data['url']}\n\n"

        content += """## Metadata

- **Type:** Security Research Paper
- **Archive:** VX Underground
- **Added to RAG:** """ + datetime.now().isoformat() + """
"""

        with open(index_path, 'w') as f:
            f.write(content)

        logger.debug(f"Created index: {index_path.name}")

    async def download_apt_reports(self, limit: int = None) -> int:
        """
        Download APT (Advanced Persistent Threat) reports

        These are comprehensive reports on APT groups, campaigns, and malware.

        Returns:
            Number of papers downloaded
        """
        logger.info("Downloading APT Reports...")

        # List of known APT report filenames (curated list)
        apt_reports = [
            "APT1.pdf",
            "APT28.pdf",
            "APT29.pdf",
            "Lazarus_Group.pdf",
            "Equation_Group.pdf",
            "Carbanak.pdf",
            "OceanLotus.pdf",
            "Turla.pdf",
            "Fancy_Bear.pdf",
            "Cozy_Bear.pdf",
            # Add more as discovered
        ]

        category = 'APT'
        category_dir = VX_DATA_DIR / category
        count = 0

        for report_name in apt_reports[:limit] if limit else apt_reports:
            # Try multiple URL patterns
            urls_to_try = [
                f"{VX_PAPERS_BASE}/APTs/{report_name}",
                f"{VX_WEBSITE_BASE}/APTs/{report_name}",
            ]

            for url in urls_to_try:
                dest_path = category_dir / self._sanitize_filename(report_name)

                file_metadata = await self.download_file(url, dest_path)

                if file_metadata:
                    # Create paper ID
                    paper_id = f"vx_apt_{count:04d}"

                    # Add metadata
                    file_metadata['category'] = category
                    file_metadata['title'] = report_name.replace('_', ' ').replace('.pdf', '')
                    file_metadata['description'] = f"APT report on {file_metadata['title']}"

                    # Save to index
                    self.vx_index['papers'][paper_id] = file_metadata
                    self.new_papers.append(paper_id)

                    # Create markdown index
                    self._create_paper_markdown(file_metadata, paper_id)

                    count += 1
                    break  # Successfully downloaded, try next report

        logger.info(f"Downloaded {count} APT reports")
        return count

    async def download_malware_analysis_papers(self, limit: int = None) -> int:
        """
        Download malware analysis papers

        Returns:
            Number of papers downloaded
        """
        logger.info("Downloading Malware Analysis Papers...")

        # Curated list of malware analysis papers
        papers = [
            "Stuxnet_Analysis.pdf",
            "WannaCry_Analysis.pdf",
            "NotPetya_Analysis.pdf",
            "Emotet_Analysis.pdf",
            "TrickBot_Analysis.pdf",
            "Ryuk_Analysis.pdf",
            "Cobalt_Strike_Analysis.pdf",
            # Add more
        ]

        category = 'Malware_Analysis'
        category_dir = VX_DATA_DIR / category
        count = 0

        for paper_name in papers[:limit] if limit else papers:
            urls_to_try = [
                f"{VX_PAPERS_BASE}/Malware%20Analysis/{paper_name}",
                f"{VX_WEBSITE_BASE}/Malware-Analysis/{paper_name}",
            ]

            for url in urls_to_try:
                dest_path = category_dir / self._sanitize_filename(paper_name)

                file_metadata = await self.download_file(url, dest_path)

                if file_metadata:
                    paper_id = f"vx_malware_{count:04d}"
                    file_metadata['category'] = category
                    file_metadata['title'] = paper_name.replace('_', ' ').replace('.pdf', '')
                    file_metadata['description'] = f"Malware analysis: {file_metadata['title']}"

                    self.vx_index['papers'][paper_id] = file_metadata
                    self.new_papers.append(paper_id)
                    self._create_paper_markdown(file_metadata, paper_id)

                    count += 1
                    break

        logger.info(f"Downloaded {count} malware analysis papers")
        return count

    async def download_technique_papers(self, limit: int = None) -> int:
        """
        Download papers on exploitation and evasion techniques

        Returns:
            Number of papers downloaded
        """
        logger.info("Downloading Technique Papers...")

        papers = [
            "Code_Injection_Techniques.pdf",
            "Anti_Debugging.pdf",
            "Anti_VM_Techniques.pdf",
            "Obfuscation_Methods.pdf",
            "Persistence_Techniques.pdf",
            "Privilege_Escalation.pdf",
            "Lateral_Movement.pdf",
            # Add more
        ]

        category = 'Techniques'
        category_dir = VX_DATA_DIR / category
        count = 0

        for paper_name in papers[:limit] if limit else papers:
            urls_to_try = [
                f"{VX_PAPERS_BASE}/Techniques/{paper_name}",
                f"{VX_WEBSITE_BASE}/Techniques/{paper_name}",
            ]

            for url in urls_to_try:
                dest_path = category_dir / self._sanitize_filename(paper_name)

                file_metadata = await self.download_file(url, dest_path)

                if file_metadata:
                    paper_id = f"vx_tech_{count:04d}"
                    file_metadata['category'] = category
                    file_metadata['title'] = paper_name.replace('_', ' ').replace('.pdf', '')
                    file_metadata['description'] = f"Technique paper: {file_metadata['title']}"

                    self.vx_index['papers'][paper_id] = file_metadata
                    self.new_papers.append(paper_id)
                    self._create_paper_markdown(file_metadata, paper_id)

                    count += 1
                    break

        logger.info(f"Downloaded {count} technique papers")
        return count

    async def download_via_git_clone(self) -> int:
        """
        Alternative method: Clone VX Underground GitHub repository

        This downloads the entire repository, which may be very large.
        Use with caution and sufficient disk space.

        Returns:
            Number of papers processed
        """
        logger.info("Cloning VX Underground Papers repository...")

        clone_dir = VX_DATA_DIR / 'vxug_git_clone'
        clone_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Clone repository (or pull if exists)
            if (clone_dir / '.git').exists():
                logger.info("Repository exists, pulling latest changes...")
                subprocess.run(
                    ['git', '-C', str(clone_dir), 'pull'],
                    check=True,
                    capture_output=True
                )
            else:
                logger.info("Cloning repository (this may take a while)...")
                subprocess.run(
                    ['git', 'clone', 'https://github.com/vxunderground/VXUG-Papers.git', str(clone_dir)],
                    check=True,
                    capture_output=True
                )

            # Index all papers in the cloned repository
            count = 0
            for pdf_file in clone_dir.rglob('*.pdf'):
                # Skip if already processed
                file_hash = self._get_file_hash(pdf_file)
                if file_hash in self.downloaded_hashes:
                    continue

                # Determine category from path
                relative_path = pdf_file.relative_to(clone_dir)
                category = 'General'
                for cat_key, cat_name in VX_CATEGORIES.items():
                    if cat_key.lower() in str(relative_path).lower():
                        category = cat_key
                        break

                # Create metadata
                paper_id = f"vx_git_{count:04d}"
                file_metadata = {
                    'filename': pdf_file.name,
                    'path': str(pdf_file),
                    'size': pdf_file.stat().st_size,
                    'hash': file_hash,
                    'category': category,
                    'title': pdf_file.stem.replace('_', ' '),
                    'description': f"Research paper from VX Underground: {pdf_file.stem}",
                    'downloaded_at': datetime.now().isoformat()
                }

                self.downloaded_hashes.add(file_hash)
                self.vx_index['papers'][paper_id] = file_metadata
                self.new_papers.append(paper_id)
                self._create_paper_markdown(file_metadata, paper_id)

                count += 1

            logger.info(f"Processed {count} papers from Git repository")
            return count

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error processing Git repository: {e}")
            return 0

    async def download_all(self, use_git: bool = False, per_category_limit: int = None):
        """
        Download papers from all categories

        Args:
            use_git: Use git clone method instead of direct downloads
            per_category_limit: Limit downloads per category (None = no limit)
        """
        logger.info("=" * 80)
        logger.info("VX Underground Paper Archive Downloader")
        logger.info("=" * 80)

        total_downloaded = 0

        if use_git:
            # Use git clone method
            total_downloaded = await self.download_via_git_clone()
        else:
            # Download by category
            total_downloaded += await self.download_apt_reports(per_category_limit)
            total_downloaded += await self.download_malware_analysis_papers(per_category_limit)
            total_downloaded += await self.download_technique_papers(per_category_limit)

        # Save index
        self._save_vx_index()

        logger.info("=" * 80)
        logger.info(f"Download Complete: {total_downloaded} new papers")
        logger.info(f"Total in archive: {len(self.vx_index['papers'])} papers")
        logger.info("=" * 80)

        return total_downloaded

    def update_rag_embeddings(self):
        """Trigger RAG system to update embeddings with VX Underground papers"""
        if not self.new_papers:
            logger.info("No new papers to add to RAG")
            return

        logger.info(f"Updating RAG embeddings with {len(self.new_papers)} new papers...")

        try:
            # Rebuild document index
            subprocess.run(
                ['python3', 'rag_system/document_processor.py'],
                check=True,
                capture_output=True
            )

            # Regenerate transformer embeddings
            subprocess.run(
                ['python3', 'rag_system/transformer_upgrade.py'],
                check=True,
                capture_output=True
            )

            logger.info("✓ RAG embeddings updated successfully")
            self.new_papers = []

        except Exception as e:
            logger.error(f"Failed to update RAG embeddings: {e}")

    def get_statistics(self) -> Dict:
        """Get download statistics"""
        papers = self.vx_index['papers']

        # Count by category
        category_counts = {}
        total_size = 0

        for paper in papers.values():
            category = paper.get('category', 'General')
            category_counts[category] = category_counts.get(category, 0) + 1
            total_size += paper.get('size', 0)

        return {
            'total_papers': len(papers),
            'total_size_mb': total_size / (1024 * 1024),
            'by_category': category_counts,
            'last_update': self.vx_index.get('last_update'),
            'pending_rag_update': len(self.new_papers)
        }


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='VX Underground Paper Archive Downloader')
    parser.add_argument(
        '--git',
        action='store_true',
        help='Use git clone method (downloads entire repository)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit downloads per category'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit'
    )
    parser.add_argument(
        '--update-rag',
        action='store_true',
        help='Force RAG update and exit'
    )

    args = parser.parse_args()

    downloader = VXUndergroundDownloader()

    if args.stats:
        # Show statistics
        stats = downloader.get_statistics()
        print("\n" + "=" * 80)
        print("VX Underground Archive Statistics")
        print("=" * 80)
        print(f"\nTotal Papers: {stats['total_papers']}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")
        print(f"Last Update: {stats['last_update']}")
        print(f"Pending RAG Update: {stats['pending_rag_update']}")

        print("\nPapers by Category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category:20s}: {count:4d}")

        print()
        return

    if args.update_rag:
        # Force RAG update
        downloader.new_papers = list(downloader.vx_index['papers'].keys())
        downloader.update_rag_embeddings()
        return

    # Download papers
    await downloader.download_all(use_git=args.git, per_category_limit=args.limit)

    # Update RAG embeddings
    downloader.update_rag_embeddings()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nDownloader stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
