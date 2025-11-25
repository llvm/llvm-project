#!/usr/bin/env python3
"""
Signal Integration for Screenshot Intelligence
Integrates Signal messages with Vector RAG via signal-cli

Features:
- signal-cli integration for message retrieval
- Automatic message ingestion to vector database
- Contact management
- Timeline correlation with screenshots
- Group message support

Integration:
- Uses signal-cli command-line tool
- Ingests into VectorRAGSystem
- Compatible with Screenshot Intelligence
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# Vector RAG
try:
    from vector_rag_system import VectorRAGSystem
except ImportError:
    VectorRAGSystem = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Signal CLI configuration"""
    signal_cli_path: str = "/usr/local/bin/signal-cli"
    phone_number: str = ""
    data_dir: Path = field(default_factory=lambda: Path.home() / ".screenshot_intel" / "signal")


class SignalIntegration:
    """
    Signal Integration for Screenshot Intelligence

    Retrieves and indexes Signal messages for timeline correlation
    """

    def __init__(
        self,
        config: SignalConfig,
        vector_rag: Optional[VectorRAGSystem] = None
    ):
        """
        Initialize Signal Integration

        Args:
            config: Signal configuration
            vector_rag: VectorRAGSystem instance (creates new if None)
        """
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Check signal-cli exists
        if not Path(config.signal_cli_path).exists():
            raise FileNotFoundError(
                f"signal-cli not found at {config.signal_cli_path}\n"
                "Install: https://github.com/AsamK/signal-cli"
            )

        # Initialize Vector RAG
        self.rag = vector_rag if vector_rag else VectorRAGSystem()

        # Contact cache
        self.contacts: Dict[str, Dict] = {}
        self.groups: Dict[str, Dict] = {}
        self.load_contacts()

        logger.info("✓ Signal Integration initialized")

    def _run_signal_cli(self, args: List[str], timeout: int = 30) -> Dict:
        """
        Run signal-cli command

        Args:
            args: Command arguments
            timeout: Command timeout

        Returns:
            Parsed JSON output
        """
        cmd = [
            self.config.signal_cli_path,
            '-u', self.config.phone_number,
            '--output', 'json'
        ] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"signal-cli error: {result.stderr}")

            # Parse JSON output (signal-cli returns JSONL)
            output = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        output.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {line}")

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"signal-cli command timed out after {timeout}s")
        except Exception as e:
            logger.error(f"signal-cli command failed: {e}")
            raise

    def load_contacts(self):
        """Load contacts and groups"""
        contacts_file = self.config.data_dir / "contacts.json"
        groups_file = self.config.data_dir / "groups.json"

        if contacts_file.exists():
            try:
                with open(contacts_file, 'r') as f:
                    self.contacts = json.load(f)
                logger.info(f"✓ Loaded {len(self.contacts)} contacts")
            except Exception as e:
                logger.warning(f"Failed to load contacts: {e}")

        if groups_file.exists():
            try:
                with open(groups_file, 'r') as f:
                    self.groups = json.load(f)
                logger.info(f"✓ Loaded {len(self.groups)} groups")
            except Exception as e:
                logger.warning(f"Failed to load groups: {e}")

    def save_contacts(self):
        """Save contacts and groups"""
        contacts_file = self.config.data_dir / "contacts.json"
        groups_file = self.config.data_dir / "groups.json"

        with open(contacts_file, 'w') as f:
            json.dump(self.contacts, f, indent=2)

        with open(groups_file, 'w') as f:
            json.dump(self.groups, f, indent=2)

    def update_contacts(self):
        """Update contact list from Signal"""
        logger.info("Updating contacts from Signal...")

        try:
            # Get contacts
            result = self._run_signal_cli(['listContacts'])

            for contact_data in result:
                if 'number' in contact_data:
                    number = contact_data['number']
                    name = contact_data.get('name', number)

                    self.contacts[number] = {
                        'number': number,
                        'name': name,
                        'profile_name': contact_data.get('profileName'),
                        'updated_at': datetime.now().isoformat()
                    }

            logger.info(f"✓ Updated {len(self.contacts)} contacts")
            self.save_contacts()

        except Exception as e:
            logger.error(f"Failed to update contacts: {e}")
            raise

    def update_groups(self):
        """Update group list from Signal"""
        logger.info("Updating groups from Signal...")

        try:
            # Get groups
            result = self._run_signal_cli(['listGroups'])

            for group_data in result:
                if 'id' in group_data:
                    group_id = group_data['id']
                    name = group_data.get('name', f"Group {group_id[:8]}")

                    self.groups[group_id] = {
                        'id': group_id,
                        'name': name,
                        'members': group_data.get('members', []),
                        'updated_at': datetime.now().isoformat()
                    }

            logger.info(f"✓ Updated {len(self.groups)} groups")
            self.save_contacts()

        except Exception as e:
            logger.error(f"Failed to update groups: {e}")
            raise

    def receive_messages(self, timeout: int = 5) -> List[Dict]:
        """
        Receive new messages from Signal

        Args:
            timeout: Receive timeout

        Returns:
            List of received messages
        """
        logger.info("Receiving messages from Signal...")

        try:
            # Receive messages
            result = self._run_signal_cli(['receive'], timeout=timeout)

            messages = []
            for msg_data in result:
                envelope = msg_data.get('envelope', {})
                data_message = envelope.get('dataMessage', {})

                # Skip non-text messages
                if not data_message.get('message'):
                    continue

                message = {
                    'timestamp': datetime.fromtimestamp(envelope.get('timestamp', 0) / 1000),
                    'source': envelope.get('source', 'unknown'),
                    'source_name': envelope.get('sourceName', 'Unknown'),
                    'group_id': data_message.get('groupInfo', {}).get('groupId'),
                    'message': data_message.get('message', ''),
                    'attachments': data_message.get('attachments', [])
                }

                messages.append(message)

            logger.info(f"✓ Received {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"Failed to receive messages: {e}")
            return []

    def ingest_message(
        self,
        message: Dict,
        chat_name: Optional[str] = None
    ) -> Dict:
        """
        Ingest a Signal message into vector database

        Args:
            message: Message data
            chat_name: Optional chat name override

        Returns:
            Ingestion result
        """
        # Determine chat name
        if chat_name is None:
            if message.get('group_id'):
                group_id = message['group_id']
                chat_name = self.groups.get(group_id, {}).get('name', f"Group {group_id[:8]}")
            else:
                source = message['source']
                chat_name = self.contacts.get(source, {}).get('name', source)

        # Additional metadata
        metadata = {
            'source_number': message['source'],
            'source_name': message.get('source_name', 'Unknown'),
            'group_id': message.get('group_id'),
            'has_attachments': len(message.get('attachments', [])) > 0,
            'attachment_count': len(message.get('attachments', []))
        }

        # Ingest into vector RAG
        result = self.rag.ingest_chat_message(
            message=message['message'],
            source='signal',
            chat_id=message.get('group_id') or message['source'],
            chat_name=chat_name,
            sender=message.get('source_name', 'Unknown'),
            timestamp=message['timestamp'],
            metadata=metadata
        )

        return result

    def sync_messages(
        self,
        max_messages: int = 1000,
        receive_timeout: int = 5
    ) -> Dict:
        """
        Sync Signal messages to vector database

        Args:
            max_messages: Maximum messages to process
            receive_timeout: Timeout for receiving messages

        Returns:
            Sync statistics
        """
        logger.info(f"Syncing Signal messages (max: {max_messages})...")

        stats = {
            'received': 0,
            'ingested': 0,
            'already_indexed': 0,
            'errors': 0
        }

        # Update contacts and groups first
        try:
            self.update_contacts()
            self.update_groups()
        except Exception as e:
            logger.warning(f"Failed to update contacts/groups: {e}")

        # Receive messages
        messages = self.receive_messages(timeout=receive_timeout)
        stats['received'] = len(messages)

        # Ingest messages
        for message in messages[:max_messages]:
            try:
                result = self.ingest_message(message)

                if result.get('status') == 'success':
                    stats['ingested'] += 1
                elif result.get('status') == 'already_indexed':
                    stats['already_indexed'] += 1
                else:
                    stats['errors'] += 1

            except Exception as e:
                logger.warning(f"Failed to ingest message: {e}")
                stats['errors'] += 1

        logger.info(f"✓ Sync complete: {stats['ingested']} new, {stats['already_indexed']} existing, {stats['errors']} errors")

        return stats

    def search_messages(
        self,
        query: str,
        limit: int = 10,
        contact: Optional[str] = None,
        group_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search Signal messages

        Args:
            query: Search query
            limit: Maximum results
            contact: Filter by contact number
            group_id: Filter by group ID

        Returns:
            List of matching messages
        """
        filters = {'source': 'signal'}

        if contact:
            filters['chat_id'] = contact
        elif group_id:
            filters['chat_id'] = group_id

        results = self.rag.search(
            query=query,
            limit=limit,
            filters=filters
        )

        messages = []
        for result in results:
            messages.append({
                'score': result.score,
                'chat_name': result.document.metadata.get('chat_name', 'Unknown'),
                'sender': result.document.metadata.get('source_name', 'Unknown'),
                'timestamp': result.document.timestamp.isoformat(),
                'text': result.document.text,
                'metadata': result.document.metadata
            })

        return messages

    def get_conversation_timeline(
        self,
        contact_or_group: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get timeline of messages for a specific conversation

        Args:
            contact_or_group: Contact number or group ID
            start_date: Start date
            end_date: End date

        Returns:
            List of messages in chronological order
        """
        events = self.rag.timeline_query(
            start_time=start_date,
            end_time=end_date,
            doc_types=['chat_message']
        )

        # Filter by chat_id
        conversation_events = [
            e for e in events
            if e.metadata.get('chat_id') == contact_or_group and e.metadata.get('source') == 'signal'
        ]

        return [{
            'timestamp': e.timestamp.isoformat(),
            'sender': e.metadata.get('source_name', 'Unknown'),
            'text': e.text,
            'metadata': e.metadata
        } for e in conversation_events]


def main():
    """Example usage"""
    print("=== Signal Integration Test ===\n")

    # Configuration
    config = SignalConfig(
        phone_number=os.getenv('SIGNAL_PHONE', '+1234567890')
    )

    if not config.phone_number:
        print("⚠️  Set SIGNAL_PHONE environment variable")
        return

    # Initialize
    signal = SignalIntegration(config)

    # Sync messages
    try:
        stats = signal.sync_messages(max_messages=100)
        print(f"\nSynced {stats['ingested']} messages")
    except Exception as e:
        print(f"Error: {e}")

    print("✓ Signal integration ready")


if __name__ == "__main__":
    main()
