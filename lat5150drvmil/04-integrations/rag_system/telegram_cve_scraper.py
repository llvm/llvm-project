#!/usr/bin/env python3
"""
Telegram CVE Scraper
Automatically scrape CVEs from t.me/cveNotify and update RAG system

Features:
- Real-time CVE monitoring
- Automatic RAG embedding updates
- CVE parsing and formatting
- Persistent storage
- Background service mode
"""

import os
import re
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import logging

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import Channel
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
CHANNELS_STR = os.getenv('SECURITY_CHANNELS', 'cveNotify')
SECURITY_CHANNELS = [ch.strip() for ch in CHANNELS_STR.split(',')]

# Configuration
SECURITY_DATA_DIR = Path('00-documentation/Security_Feed')
SECURITY_INDEX_FILE = Path('rag_system/security_index.json')
SESSION_FILE = 'telegram_cve_session'

# Auto-update settings
AUTO_UPDATE_EMBEDDINGS = os.getenv('AUTO_UPDATE_EMBEDDINGS', 'true').lower() == 'true'
UPDATE_BATCH_SIZE = int(os.getenv('UPDATE_BATCH_SIZE', '10'))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL_SECONDS', '300'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system/cve_scraper.log'),
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
        """
        Parse CVE information from Telegram message

        Returns:
            Dict with CVE details
        """
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
            'scraped_at': datetime.now().isoformat()
        }


class CVEScraper:
    """Scrape CVEs from Telegram and update RAG system"""

    def __init__(self):
        """Initialize Telegram CVE scraper"""
        if not API_ID or not API_HASH:
            raise ValueError(
                "Telegram credentials not found!\n"
                "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env.telegram"
            )

        self.client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
        self.cve_index = self._load_cve_index()
        self.new_cves = []
        self.parser = CVEParser()

        # Create CVE data directory
        CVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("CVE Scraper initialized")

    def _load_cve_index(self) -> Dict:
        """Load existing CVE index"""
        if CVE_INDEX_FILE.exists():
            with open(CVE_INDEX_FILE, 'r') as f:
                return json.load(f)
        return {'cves': {}, 'last_update': None}

    def _save_cve_index(self):
        """Save CVE index"""
        self.cve_index['last_update'] = datetime.now().isoformat()
        with open(CVE_INDEX_FILE, 'w') as f:
            json.dump(self.cve_index, f, indent=2)

    def _save_cve_to_file(self, cve_data: Dict):
        """Save CVE to individual markdown file"""
        cve_id = cve_data['cve_id']
        filename = CVE_DATA_DIR / f"{cve_id}.md"

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

        logger.info(f"Saved {cve_id} to {filename}")

    def _update_rag_embeddings(self):
        """Trigger RAG system to update embeddings with new CVEs"""
        if not AUTO_UPDATE_EMBEDDINGS or not self.new_cves:
            return

        logger.info(f"Updating RAG embeddings with {len(self.new_cves)} new CVEs...")

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
            self.new_cves = []

        except Exception as e:
            logger.error(f"Failed to update RAG embeddings: {e}")

    async def scrape_channel_history(self, channel_name: str, limit: int = 100):
        """
        Scrape historical messages from a security channel

        Args:
            channel_name: Channel username (without @)
            limit: Number of messages to scrape
        """
        logger.info(f"Scraping {limit} messages from @{channel_name}...")

        try:
            channel = await self.client.get_entity(channel_name)

            count = 0
            async for message in self.client.iter_messages(channel, limit=limit):
                if message.text:
                    cve_data = self.parser.parse_cve_message(message.text)

                    if cve_data and cve_data['cve_id']:
                        cve_id = cve_data['cve_id']

                        # Skip if already indexed
                        if cve_id in self.cve_index['cves']:
                            continue

                        # Add channel source
                        cve_data['source_channel'] = channel_name

                        # Save CVE
                        self._save_cve_to_file(cve_data)
                        self.cve_index['cves'][cve_id] = cve_data
                        self.new_cves.append(cve_id)
                        count += 1

                        logger.info(f"[@{channel_name}] Found new CVE: {cve_id} ({cve_data.get('severity', 'Unknown')})")

            logger.info(f"[@{channel_name}] Scraped {count} new CVEs from history")

            # Save index
            self._save_cve_index()

            # Update RAG if we have enough new CVEs
            if len(self.new_cves) >= UPDATE_BATCH_SIZE:
                self._update_rag_embeddings()

        except Exception as e:
            logger.error(f"[@{channel_name}] Error scraping channel history: {e}")

    async def scrape_all_channels_history(self, limit: int = 100):
        """
        Scrape historical messages from all configured security channels

        Args:
            limit: Number of messages to scrape per channel
        """
        logger.info(f"Scraping {len(SECURITY_CHANNELS)} security channels...")
        logger.info(f"Channels: {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")

        for channel in SECURITY_CHANNELS:
            await self.scrape_channel_history(channel, limit)

        logger.info(f"Completed scraping all {len(SECURITY_CHANNELS)} channels")

    async def monitor_new_cves(self):
        """Monitor all configured security channels for new CVE messages in real-time"""
        logger.info(f"Starting real-time monitoring of {len(SECURITY_CHANNELS)} security channels...")
        logger.info(f"Monitoring: {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")

        @self.client.on(events.NewMessage(chats=SECURITY_CHANNELS))
        async def handler(event):
            message = event.message.text
            channel_name = event.chat.username if hasattr(event.chat, 'username') else 'unknown'

            if message:
                cve_data = self.parser.parse_cve_message(message)

                if cve_data and cve_data['cve_id']:
                    cve_id = cve_data['cve_id']

                    # Skip if already indexed
                    if cve_id in self.cve_index['cves']:
                        logger.debug(f"[@{channel_name}] Skipping duplicate: {cve_id}")
                        return

                    # Add channel source
                    cve_data['source_channel'] = channel_name

                    # Save new CVE
                    logger.info(f"[@{channel_name}] 🆕 New CVE detected: {cve_id} ({cve_data.get('severity', 'Unknown')})")

                    self._save_cve_to_file(cve_data)
                    self.cve_index['cves'][cve_id] = cve_data
                    self.new_cves.append(cve_id)
                    self._save_cve_index()

                    # Update RAG if batch size reached
                    if len(self.new_cves) >= UPDATE_BATCH_SIZE:
                        self._update_rag_embeddings()

        # Keep running
        await self.client.run_until_disconnected()

    async def run_periodic_update(self):
        """Run periodic updates (for batch processing)"""
        while True:
            await asyncio.sleep(UPDATE_INTERVAL)

            if self.new_cves:
                logger.info(f"Periodic update: {len(self.new_cves)} CVEs pending")
                self._update_rag_embeddings()

    async def start(self, scrape_history: bool = True, monitor_realtime: bool = True, oneshot: bool = False):
        """
        Start CVE scraper

        Args:
            scrape_history: Scrape historical messages first
            monitor_realtime: Monitor for new messages
            oneshot: Run once and exit (for timer-based execution)
        """
        await self.client.start()

        logger.info("=" * 70)
        logger.info("Telegram Security Feed Scraper Started")
        logger.info("=" * 70)
        logger.info(f"Channels ({len(SECURITY_CHANNELS)}): {', '.join('@' + ch for ch in SECURITY_CHANNELS)}")
        logger.info(f"Auto-update embeddings: {AUTO_UPDATE_EMBEDDINGS}")
        logger.info(f"Batch size: {UPDATE_BATCH_SIZE}")
        logger.info(f"Mode: {'One-shot' if oneshot else 'Continuous'}")
        logger.info("=" * 70)

        # Scrape historical messages from all channels
        if scrape_history:
            await self.scrape_all_channels_history(limit=100)

        # One-shot mode: update and exit
        if oneshot:
            if self.new_cves:
                logger.info(f"One-shot: Processing {len(self.new_cves)} new CVEs")
                self._update_rag_embeddings()
            logger.info("One-shot mode: Exiting")
            await self.client.disconnect()
            return

        # Start monitoring (continuous mode)
        if monitor_realtime:
            # Start periodic updater
            asyncio.create_task(self.run_periodic_update())

            # Monitor new messages
            await self.monitor_new_cves()

    def get_statistics(self) -> Dict:
        """Get CVE scraper statistics"""
        cves = self.cve_index['cves']

        # Count by severity
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

        return {
            'total_cves': len(cves),
            'by_severity': severity_counts,
            'last_update': self.cve_index.get('last_update'),
            'pending_rag_update': len(self.new_cves)
        }


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Telegram CVE Scraper')
    parser.add_argument(
        '--history',
        type=int,
        default=100,
        help='Number of historical messages to scrape'
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

    scraper = CVEScraper()

    if args.stats:
        # Show statistics
        stats = scraper.get_statistics()
        print("\n" + "=" * 70)
        print("CVE Scraper Statistics")
        print("=" * 70)
        print(f"\nTotal CVEs: {stats['total_cves']}")
        print(f"Last Update: {stats['last_update']}")
        print(f"Pending RAG Update: {stats['pending_rag_update']}")
        print("\nBy Severity:")
        for severity, count in stats['by_severity'].items():
            print(f"  {severity:10s}: {count:4d}")
        print()
        return

    if args.update_rag:
        # Force RAG update
        scraper.new_cves = list(scraper.cve_index['cves'].keys())
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
