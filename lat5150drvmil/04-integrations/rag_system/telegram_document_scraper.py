#!/usr/bin/env python3
"""
Enhanced Telegram Document Scraper
Scrapes CVEs, documentation, and files from multiple Telegram security channels

Features:
- Multi-channel monitoring (cveNotify, Pwn3rzs, custom channels)
- File attachment downloads (.md, .pdf, .txt, etc.)
- CVE parsing and general document handling
- Automatic RAG embedding updates
- Real-time monitoring and historical scraping
"""

import os
import re
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Set
import logging
import hashlib
import mimetypes

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import Channel, DocumentAttributeFilename
    from dotenv import load_dotenv
    TELETHON_AVAILABLE = True
except ImportError:
    print("❌ Telethon not installed!")
    print("Install with: pip install telethon python-dotenv")
    TELETHON_AVAILABLE = False
    import sys
    sys.exit(1)

# Load environment variables
load_dotenv('.env.telegram')

# Telegram credentials
API_ID = int(os.getenv('TELEGRAM_API_ID', '0'))
API_HASH = os.getenv('TELEGRAM_API_HASH', '')

# Channels to scrape (comma-separated)
# Default includes cveNotify and Pwn3rzs
DEFAULT_CHANNELS = 'cveNotify,Pwn3rzs'
CHANNELS_STR = os.getenv('SECURITY_CHANNELS', DEFAULT_CHANNELS)
SECURITY_CHANNELS = [ch.strip() for ch in CHANNELS_STR.split(',')]

# Configuration
SECURITY_DATA_DIR = Path('00-documentation/Security_Feed')
PWNER_DATA_DIR = Path('00-documentation/Security_Feed/Pwn3rzs')
DOWNLOADS_DIR = Path('00-documentation/Security_Feed/Downloads')
SECURITY_INDEX_FILE = Path('rag_system/security_index.json')
SESSION_FILE = 'telegram_document_session'

# File download settings
ALLOWED_EXTENSIONS = {'.md', '.pdf', '.txt', '.doc', '.docx', '.json', '.yaml', '.yml', '.sh', '.py'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Auto-update settings
AUTO_UPDATE_EMBEDDINGS = os.getenv('AUTO_UPDATE_EMBEDDINGS', 'true').lower() == 'true'
UPDATE_BATCH_SIZE = int(os.getenv('UPDATE_BATCH_SIZE', '10'))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL_SECONDS', '300'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/document_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CVEParser:
    """Parse CVE information from Telegram messages"""

    CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE)
    CVSS_PATTERN = re.compile(r'CVSS[:\s]+(\d+\.?\d*)', re.IGNORECASE)
    SEVERITY_PATTERN = re.compile(r'Severity[:\s]+(Critical|High|Medium|Low)', re.IGNORECASE)

    @staticmethod
    def extract_cve_id(text: str) -> Optional[str]:
        """Extract CVE ID from text"""
        match = CVEParser.CVE_PATTERN.search(text)
        return match.group(0).upper() if match else None

    @staticmethod
    def extract_cvss_score(text: str) -> Optional[float]:
        """Extract CVSS score from text"""
        match = CVEParser.CVSS_PATTERN.search(text)
        return float(match.group(1)) if match else None

    @staticmethod
    def extract_severity(text: str) -> Optional[str]:
        """Extract severity level from text"""
        match = CVEParser.SEVERITY_PATTERN.search(text)
        return match.group(1).capitalize() if match else None

    @staticmethod
    def parse_cve_message(message: str) -> Dict:
        """Parse CVE information from Telegram message"""
        cve_id = CVEParser.extract_cve_id(message)

        if not cve_id:
            return {}

        # Extract other fields
        cvss_score = CVEParser.extract_cvss_score(message)
        severity = CVEParser.extract_severity(message)

        # Infer severity from CVSS if not explicit
        if not severity and cvss_score:
            if cvss_score >= 9.0:
                severity = "Critical"
            elif cvss_score >= 7.0:
                severity = "High"
            elif cvss_score >= 4.0:
                severity = "Medium"
            else:
                severity = "Low"

        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', message)

        return {
            'cve_id': cve_id,
            'cvss_score': cvss_score,
            'severity': severity,
            'description': message.strip(),
            'urls': urls,
            'scraped_at': datetime.now().isoformat(),
            'type': 'cve'
        }


class DocumentParser:
    """Parse general security documentation from Telegram messages"""

    @staticmethod
    def parse_document_message(message: str, channel_name: str) -> Dict:
        """
        Parse general documentation message

        Returns:
            Dict with document metadata
        """
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', message)

        # Detect keywords for categorization
        categories = []
        keywords = {
            'exploit': ['exploit', 'poc', 'proof of concept', '0day', 'zero-day'],
            'malware': ['malware', 'ransomware', 'trojan', 'virus', 'backdoor'],
            'vulnerability': ['vulnerability', 'vuln', 'bug', 'flaw', 'weakness'],
            'research': ['research', 'paper', 'analysis', 'study', 'whitepaper'],
            'tool': ['tool', 'framework', 'utility', 'script'],
            'forensics': ['forensics', 'dfir', 'incident response'],
            'reversing': ['reverse engineering', 'reversing', 'disassembly', 'decompile']
        }

        message_lower = message.lower()
        for category, terms in keywords.items():
            if any(term in message_lower for term in terms):
                categories.append(category)

        if not categories:
            categories = ['general']

        return {
            'description': message.strip(),
            'urls': urls,
            'categories': categories,
            'scraped_at': datetime.now().isoformat(),
            'source_channel': channel_name,
            'type': 'document'
        }


class EnhancedSecurityScraper:
    """Scrape CVEs, documents, and files from Telegram security channels"""

    def __init__(self):
        """Initialize enhanced Telegram security scraper"""
        if not API_ID or not API_HASH:
            raise ValueError(
                "Telegram credentials not found!\n"
                "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env.telegram"
            )

        self.client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
        self.security_index = self._load_security_index()
        self.new_documents = []
        self.cve_parser = CVEParser()
        self.doc_parser = DocumentParser()
        self.downloaded_files: Set[str] = set()

        # Create data directories
        SECURITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PWNER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Load downloaded files index
        self._load_downloaded_files()

        logger.info("Enhanced Security Scraper initialized")
        logger.info(f"Monitoring channels: {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")

    def _load_security_index(self) -> Dict:
        """Load existing security document index"""
        if SECURITY_INDEX_FILE.exists():
            with open(SECURITY_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {
            'cves': {},
            'documents': {},
            'files': {},
            'last_update': None
        }

    def _save_security_index(self):
        """Save security document index"""
        self.security_index['last_update'] = datetime.now().isoformat()
        with open(SECURITY_INDEX_FILE, 'w') as f:
            json.dump(self.security_index, f, indent=2)

    def _load_downloaded_files(self):
        """Load set of already downloaded file hashes"""
        files_data = self.security_index.get('files', {})
        self.downloaded_files = set(files_data.keys())

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _save_cve_to_file(self, cve_data: Dict):
        """Save CVE to individual markdown file"""
        cve_id = cve_data['cve_id']
        filename = SECURITY_DATA_DIR / f"{cve_id}.md"

        content = f"""# {cve_id}

**Severity:** {cve_data.get('severity', 'Unknown')}
**CVSS Score:** {cve_data.get('cvss_score', 'N/A')}
**Discovered:** {cve_data['scraped_at']}

## Description

{cve_data['description']}

## References

"""
        if cve_data.get('urls'):
            for url in cve_data['urls']:
                content += f"- {url}\n"
        else:
            content += "- No references available\n"

        source_channel = cve_data.get('source_channel', 'unknown')
        content += f"""

## Metadata

- **Source:** Telegram @{source_channel}
- **Added to RAG:** {datetime.now().isoformat()}
- **Category:** Security / CVE
"""

        with open(filename, 'w') as f:
            f.write(content)

        logger.info(f"Saved CVE {cve_id} to {filename}")

    def _save_document_to_file(self, doc_data: Dict, doc_id: str):
        """Save general document to markdown file"""
        channel = doc_data.get('source_channel', 'unknown')

        # Choose directory based on channel
        if channel.lower() == 'pwn3rzs':
            base_dir = PWNER_DATA_DIR
        else:
            base_dir = SECURITY_DATA_DIR

        filename = base_dir / f"{doc_id}.md"

        categories_str = ', '.join(doc_data.get('categories', ['general']))

        content = f"""# Security Document - {doc_id}

**Source:** @{channel}
**Categories:** {categories_str}
**Scraped:** {doc_data['scraped_at']}

## Content

{doc_data['description']}

"""
        if doc_data.get('urls'):
            content += "## References\n\n"
            for url in doc_data['urls']:
                content += f"- {url}\n"

        if doc_data.get('file_path'):
            content += f"\n## Attached File\n\n"
            content += f"- File: `{doc_data['file_path']}`\n"
            content += f"- Hash: {doc_data.get('file_hash', 'N/A')}\n"

        content += f"""

## Metadata

- **Type:** Security Documentation
- **Added to RAG:** {datetime.now().isoformat()}
"""

        with open(filename, 'w') as f:
            f.write(content)

        logger.info(f"Saved document {doc_id} to {filename}")

    async def _download_file(self, message, channel_name: str) -> Optional[Dict]:
        """
        Download file attachment from message

        Returns:
            Dict with file metadata if successful
        """
        if not message.media or not hasattr(message.media, 'document'):
            return None

        document = message.media.document

        # Get filename from attributes
        filename = None
        for attr in document.attributes:
            if isinstance(attr, DocumentAttributeFilename):
                filename = attr.file_name
                break

        if not filename:
            # Generate filename from document ID
            ext = mimetypes.guess_extension(document.mime_type) or ''
            filename = f"document_{document.id}{ext}"

        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.debug(f"Skipping file with extension {file_ext}: {filename}")
            return None

        # Check file size
        if document.size > MAX_FILE_SIZE:
            logger.warning(f"File too large ({document.size} bytes): {filename}")
            return None

        # Download file
        try:
            # Create channel-specific subdirectory
            channel_dir = DOWNLOADS_DIR / channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)

            download_path = channel_dir / filename

            # Skip if already exists
            if download_path.exists():
                logger.debug(f"File already exists: {download_path}")
                return None

            logger.info(f"[@{channel_name}] Downloading: {filename} ({document.size} bytes)")

            await self.client.download_media(message.media, download_path)

            # Calculate hash
            file_hash = self._get_file_hash(download_path)

            # Check if we already have this file (by hash)
            if file_hash in self.downloaded_files:
                logger.info(f"Duplicate file (hash match): {filename}")
                download_path.unlink()  # Delete duplicate
                return None

            self.downloaded_files.add(file_hash)

            file_metadata = {
                'filename': filename,
                'path': str(download_path),
                'size': document.size,
                'mime_type': document.mime_type,
                'hash': file_hash,
                'channel': channel_name,
                'downloaded_at': datetime.now().isoformat()
            }

            logger.info(f"[@{channel_name}] ✓ Downloaded: {filename}")

            return file_metadata

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None

    def _update_rag_embeddings(self):
        """Trigger RAG system to update embeddings with new documents"""
        if not AUTO_UPDATE_EMBEDDINGS or not self.new_documents:
            return

        logger.info(f"Updating RAG embeddings with {len(self.new_documents)} new documents...")

        try:
            # Rebuild document index
            import subprocess
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
            self.new_documents = []

        except Exception as e:
            logger.error(f"Failed to update RAG embeddings: {e}")

    async def scrape_channel_history(self, channel_name: str, limit: int = 100):
        """
        Scrape historical messages and files from a security channel

        Args:
            channel_name: Channel username (without @)
            limit: Number of messages to scrape
        """
        logger.info(f"[@{channel_name}] Scraping {limit} messages...")

        try:
            channel = await self.client.get_entity(channel_name)

            cve_count = 0
            doc_count = 0
            file_count = 0

            async for message in self.client.iter_messages(channel, limit=limit):
                # Download file attachments
                if message.media:
                    file_metadata = await self._download_file(message, channel_name)
                    if file_metadata:
                        file_hash = file_metadata['hash']
                        self.security_index['files'][file_hash] = file_metadata
                        file_count += 1

                # Process text message
                if message.text:
                    # Try CVE parsing first
                    cve_data = self.cve_parser.parse_cve_message(message.text)

                    if cve_data and cve_data.get('cve_id'):
                        cve_id = cve_data['cve_id']

                        # Skip if already indexed
                        if cve_id in self.security_index['cves']:
                            continue

                        # Add channel source
                        cve_data['source_channel'] = channel_name

                        # Save CVE
                        self._save_cve_to_file(cve_data)
                        self.security_index['cves'][cve_id] = cve_data
                        self.new_documents.append(cve_id)
                        cve_count += 1

                        logger.info(f"[@{channel_name}] Found CVE: {cve_id} ({cve_data.get('severity', 'Unknown')})")

                    else:
                        # Parse as general document
                        doc_data = self.doc_parser.parse_document_message(message.text, channel_name)

                        # Create document ID
                        doc_id = f"{channel_name}_{message.id}"

                        # Skip if already indexed
                        if doc_id in self.security_index['documents']:
                            continue

                        # Attach file metadata if downloaded
                        if message.media and file_count > 0:
                            # Link to last downloaded file
                            last_file_hash = list(self.security_index['files'].keys())[-1] if self.security_index['files'] else None
                            if last_file_hash:
                                doc_data['file_path'] = self.security_index['files'][last_file_hash]['path']
                                doc_data['file_hash'] = last_file_hash

                        # Save document
                        self._save_document_to_file(doc_data, doc_id)
                        self.security_index['documents'][doc_id] = doc_data
                        self.new_documents.append(doc_id)
                        doc_count += 1

                        logger.debug(f"[@{channel_name}] Found document: {doc_id}")

            logger.info(f"[@{channel_name}] Scraped: {cve_count} CVEs, {doc_count} documents, {file_count} files")

            # Save index
            self._save_security_index()

            # Update RAG if we have enough new documents
            if len(self.new_documents) >= UPDATE_BATCH_SIZE:
                self._update_rag_embeddings()

        except Exception as e:
            logger.error(f"[@{channel_name}] Error scraping channel: {e}")

    async def scrape_all_channels_history(self, limit: int = 100):
        """Scrape historical messages from all configured security channels"""
        logger.info(f"Scraping {len(SECURITY_CHANNELS)} security channels...")
        logger.info(f"Channels: {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")

        for channel in SECURITY_CHANNELS:
            await self.scrape_channel_history(channel, limit)

        logger.info(f"Completed scraping all {len(SECURITY_CHANNELS)} channels")

    async def monitor_new_messages(self):
        """Monitor all configured security channels for new messages and files"""
        logger.info(f"Starting real-time monitoring of {len(SECURITY_CHANNELS)} channels...")
        logger.info(f"Monitoring: {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")

        @self.client.on(events.NewMessage(chats=SECURITY_CHANNELS))
        async def handler(event):
            message = event.message
            channel_name = event.chat.username if hasattr(event.chat, 'username') else 'unknown'

            # Download file attachments
            if message.media:
                file_metadata = await self._download_file(message, channel_name)
                if file_metadata:
                    file_hash = file_metadata['hash']
                    self.security_index['files'][file_hash] = file_metadata
                    self._save_security_index()
                    logger.info(f"[@{channel_name}] 📎 New file: {file_metadata['filename']}")

            # Process text message
            if message.text:
                # Try CVE parsing first
                cve_data = self.cve_parser.parse_cve_message(message.text)

                if cve_data and cve_data.get('cve_id'):
                    cve_id = cve_data['cve_id']

                    # Skip if already indexed
                    if cve_id in self.security_index['cves']:
                        return

                    # Add channel source
                    cve_data['source_channel'] = channel_name

                    # Save new CVE
                    logger.info(f"[@{channel_name}] 🆕 New CVE: {cve_id} ({cve_data.get('severity', 'Unknown')})")

                    self._save_cve_to_file(cve_data)
                    self.security_index['cves'][cve_id] = cve_data
                    self.new_documents.append(cve_id)
                    self._save_security_index()

                else:
                    # Parse as general document
                    doc_data = self.doc_parser.parse_document_message(message.text, channel_name)
                    doc_id = f"{channel_name}_{message.id}"

                    # Skip if already indexed
                    if doc_id in self.security_index['documents']:
                        return

                    # Save document
                    logger.info(f"[@{channel_name}] 📄 New document: {', '.join(doc_data['categories'])}")

                    self._save_document_to_file(doc_data, doc_id)
                    self.security_index['documents'][doc_id] = doc_data
                    self.new_documents.append(doc_id)
                    self._save_security_index()

                # Update RAG if batch size reached
                if len(self.new_documents) >= UPDATE_BATCH_SIZE:
                    self._update_rag_embeddings()

        # Keep running
        await self.client.run_until_disconnected()

    async def run_periodic_update(self):
        """Run periodic RAG updates (for batch processing)"""
        while True:
            await asyncio.sleep(UPDATE_INTERVAL)

            if self.new_documents:
                logger.info(f"Periodic update: {len(self.new_documents)} documents pending")
                self._update_rag_embeddings()

    async def start(self, scrape_history: bool = True, monitor_realtime: bool = True, oneshot: bool = False):
        """
        Start enhanced security scraper

        Args:
            scrape_history: Scrape historical messages first
            monitor_realtime: Monitor for new messages
            oneshot: Run once and exit (for timer-based execution)
        """
        await self.client.start()

        logger.info("=" * 80)
        logger.info("Enhanced Telegram Security Scraper Started")
        logger.info("=" * 80)
        logger.info(f"Channels ({len(SECURITY_CHANNELS)}): {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")
        logger.info(f"Auto-update embeddings: {AUTO_UPDATE_EMBEDDINGS}")
        logger.info(f"Batch size: {UPDATE_BATCH_SIZE}")
        logger.info(f"Mode: {'One-shot' if oneshot else 'Continuous'}")
        logger.info(f"File downloads: {', '.join(ALLOWED_EXTENSIONS)}")
        logger.info("=" * 80)

        # Scrape historical messages from all channels
        if scrape_history:
            await self.scrape_all_channels_history(limit=200)

        # One-shot mode: update and exit
        if oneshot:
            if self.new_documents:
                logger.info(f"One-shot: Processing {len(self.new_documents)} new documents")
                self._update_rag_embeddings()
            logger.info("One-shot mode: Exiting")
            await self.client.disconnect()
            return

        # Start monitoring (continuous mode)
        if monitor_realtime:
            # Start periodic updater
            asyncio.create_task(self.run_periodic_update())

            # Monitor new messages
            await self.monitor_new_messages()

    def get_statistics(self) -> Dict:
        """Get scraper statistics"""
        cves = self.security_index['cves']
        documents = self.security_index['documents']
        files = self.security_index['files']

        # Count CVEs by severity
        severity_counts = {
            'Critical': 0,
            'High': 0,
            'Medium': 0,
            'Low': 0,
            'Unknown': 0
        }

        for cve_data in cves.values():
            severity = cve_data.get('severity', 'Unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count documents by category
        category_counts = {}
        for doc_data in documents.values():
            for category in doc_data.get('categories', ['general']):
                category_counts[category] = category_counts.get(category, 0) + 1

        # Count files by extension
        extension_counts = {}
        for file_data in files.values():
            ext = Path(file_data['filename']).suffix.lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

        return {
            'total_cves': len(cves),
            'total_documents': len(documents),
            'total_files': len(files),
            'by_severity': severity_counts,
            'by_category': category_counts,
            'by_extension': extension_counts,
            'last_update': self.security_index.get('last_update'),
            'pending_rag_update': len(self.new_documents)
        }


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Telegram Security Scraper')
    parser.add_argument(
        '--history',
        type=int,
        default=200,
        help='Number of historical messages to scrape per channel'
    )
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Disable real-time monitoring'
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
    parser.add_argument(
        '--oneshot',
        action='store_true',
        help='Run once and exit (for systemd timer)'
    )

    args = parser.parse_args()

    scraper = EnhancedSecurityScraper()

    if args.stats:
        # Show statistics
        stats = scraper.get_statistics()
        print("\n" + "=" * 80)
        print("Enhanced Security Scraper Statistics")
        print("=" * 80)
        print(f"\nTotal CVEs: {stats['total_cves']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Files: {stats['total_files']}")
        print(f"Last Update: {stats['last_update']}")
        print(f"Pending RAG Update: {stats['pending_rag_update']}")

        print("\nCVEs by Severity:")
        for severity, count in stats['by_severity'].items():
            print(f"  {severity:10s}: {count:4d}")

        print("\nDocuments by Category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category:20s}: {count:4d}")

        print("\nFiles by Extension:")
        for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ext:10s}: {count:4d}")

        print()
        return

    if args.update_rag:
        # Force RAG update
        scraper.new_documents = list(scraper.security_index['cves'].keys())
        scraper.new_documents.extend(list(scraper.security_index['documents'].keys()))
        scraper._update_rag_embeddings()
        return

    # Start scraper
    await scraper.start(
        scrape_history=True,
        monitor_realtime=not args.no_monitor and not args.oneshot,
        oneshot=args.oneshot
    )


if __name__ == '__main__':
    if not TELETHON_AVAILABLE:
        print("Install required packages:")
        print("  pip install telethon python-dotenv")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nScraper stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
