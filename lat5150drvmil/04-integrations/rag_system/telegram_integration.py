#!/usr/bin/env python3
"""
Telegram Integration for Screenshot Intelligence
Integrates Telegram messages with Vector RAG for timeline correlation

Features:
- Telethon API integration for message retrieval
- Automatic message ingestion to vector database
- Chat history synchronization
- Timeline correlation with screenshots
- Entity extraction (users, channels, media)

Integration:
- Uses existing telegram_document_scraper.py infrastructure
- Ingests into VectorRAGSystem
- Compatible with Screenshot Intelligence
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# Telethon for Telegram API
try:
    from telethon import TelegramClient, events
    from telethon.tl.types import User, Channel, Chat, Message
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
    print("⚠️  Telethon not installed. Run: pip install telethon")

# Vector RAG
try:
    from vector_rag_system import VectorRAGSystem
except ImportError:
    VectorRAGSystem = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram API configuration"""
    api_id: str
    api_hash: str
    phone_number: str
    session_name: str = "screenshot_intel_session"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".screenshot_intel" / "telegram")


class TelegramIntegration:
    """
    Telegram Integration for Screenshot Intelligence

    Retrieves and indexes Telegram messages for timeline correlation
    """

    def __init__(
        self,
        config: TelegramConfig,
        vector_rag: Optional[VectorRAGSystem] = None
    ):
        """
        Initialize Telegram Integration

        Args:
            config: Telegram configuration
            vector_rag: VectorRAGSystem instance (creates new if None)
        """
        if not TELETHON_AVAILABLE:
            raise ImportError("Telethon required. Install: pip install telethon")

        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vector RAG
        self.rag = vector_rag if vector_rag else VectorRAGSystem()

        # Initialize Telegram client
        session_path = self.config.data_dir / config.session_name
        self.client = TelegramClient(
            str(session_path),
            config.api_id,
            config.api_hash
        )

        # Chat cache
        self.monitored_chats: Dict[str, Dict] = {}
        self.load_monitored_chats()

        logger.info("✓ Telegram Integration initialized")

    def load_monitored_chats(self):
        """Load list of monitored chats"""
        chats_file = self.config.data_dir / "monitored_chats.json"
        if chats_file.exists():
            try:
                with open(chats_file, 'r') as f:
                    self.monitored_chats = json.load(f)
                logger.info(f"✓ Loaded {len(self.monitored_chats)} monitored chats")
            except Exception as e:
                logger.warning(f"Failed to load monitored chats: {e}")

    def save_monitored_chats(self):
        """Save list of monitored chats"""
        chats_file = self.config.data_dir / "monitored_chats.json"
        with open(chats_file, 'w') as f:
            json.dump(self.monitored_chats, f, indent=2)

    async def start(self):
        """Start Telegram client"""
        await self.client.start(phone=self.config.phone_number)
        logger.info("✓ Telegram client started")

    async def stop(self):
        """Stop Telegram client"""
        await self.client.disconnect()
        logger.info("✓ Telegram client stopped")

    async def add_monitored_chat(
        self,
        chat_identifier: str,
        chat_name: Optional[str] = None
    ):
        """
        Add a chat to monitor

        Args:
            chat_identifier: Chat username, phone, or ID
            chat_name: Optional display name
        """
        try:
            entity = await self.client.get_entity(chat_identifier)

            chat_id = str(entity.id)
            if chat_name is None:
                if hasattr(entity, 'title'):
                    chat_name = entity.title
                elif hasattr(entity, 'username'):
                    chat_name = entity.username
                else:
                    chat_name = f"Chat {chat_id}"

            self.monitored_chats[chat_id] = {
                'chat_id': chat_id,
                'chat_name': chat_name,
                'chat_identifier': chat_identifier,
                'added_at': datetime.now().isoformat(),
                'last_sync': None,
                'message_count': 0
            }

            self.save_monitored_chats()
            logger.info(f"✓ Added monitored chat: {chat_name} ({chat_id})")

        except Exception as e:
            logger.error(f"Failed to add monitored chat {chat_identifier}: {e}")
            raise

    async def sync_chat_history(
        self,
        chat_id: str,
        limit: int = 1000,
        offset_date: Optional[datetime] = None
    ) -> Dict:
        """
        Sync chat history to vector database

        Args:
            chat_id: Chat ID to sync
            limit: Maximum messages to retrieve
            offset_date: Start from this date (None = from beginning)

        Returns:
            Sync statistics
        """
        if chat_id not in self.monitored_chats:
            raise ValueError(f"Chat {chat_id} not monitored")

        chat_info = self.monitored_chats[chat_id]
        chat_identifier = chat_info['chat_identifier']

        logger.info(f"Syncing chat: {chat_info['chat_name']} (limit: {limit})")

        stats = {
            'chat_id': chat_id,
            'chat_name': chat_info['chat_name'],
            'retrieved': 0,
            'ingested': 0,
            'already_indexed': 0,
            'errors': 0
        }

        try:
            entity = await self.client.get_entity(chat_identifier)

            # Retrieve messages
            messages = []
            async for message in self.client.iter_messages(
                entity,
                limit=limit,
                offset_date=offset_date
            ):
                messages.append(message)
                stats['retrieved'] += 1

            logger.info(f"Retrieved {len(messages)} messages")

            # Ingest messages
            for message in messages:
                try:
                    result = await self._ingest_message(message, chat_info)

                    if result.get('status') == 'success':
                        stats['ingested'] += 1
                    elif result.get('status') == 'already_indexed':
                        stats['already_indexed'] += 1
                    else:
                        stats['errors'] += 1

                except Exception as e:
                    logger.warning(f"Failed to ingest message {message.id}: {e}")
                    stats['errors'] += 1

            # Update chat info
            chat_info['last_sync'] = datetime.now().isoformat()
            chat_info['message_count'] += stats['ingested']
            self.save_monitored_chats()

            logger.info(f"✓ Sync complete: {stats['ingested']} new, {stats['already_indexed']} existing, {stats['errors']} errors")

        except Exception as e:
            logger.error(f"Failed to sync chat {chat_id}: {e}")
            raise

        return stats

    async def _ingest_message(
        self,
        message: Message,
        chat_info: Dict
    ) -> Dict:
        """
        Ingest a single Telegram message

        Args:
            message: Telegram message
            chat_info: Chat information

        Returns:
            Ingestion result
        """
        # Skip messages without text
        if not message.text:
            return {'status': 'skipped', 'reason': 'no_text'}

        # Extract sender info
        sender_id = str(message.sender_id) if message.sender_id else 'unknown'
        sender_name = 'Unknown'

        if message.sender:
            if hasattr(message.sender, 'first_name'):
                sender_name = message.sender.first_name
                if hasattr(message.sender, 'last_name') and message.sender.last_name:
                    sender_name += f" {message.sender.last_name}"
            elif hasattr(message.sender, 'title'):
                sender_name = message.sender.title

        # Additional metadata
        metadata = {
            'message_id': message.id,
            'sender_id': sender_id,
            'sender_name': sender_name,
            'has_media': message.media is not None,
            'is_reply': message.reply_to is not None,
            'views': message.views if hasattr(message, 'views') else None,
            'forwards': message.forwards if hasattr(message, 'forwards') else None
        }

        # Extract entities (mentions, hashtags, URLs)
        if message.entities:
            entities = []
            for entity in message.entities:
                entity_type = entity.__class__.__name__
                entities.append({
                    'type': entity_type,
                    'offset': entity.offset,
                    'length': entity.length
                })
            metadata['entities'] = entities

        # Ingest into vector RAG
        result = self.rag.ingest_chat_message(
            message=message.text,
            source='telegram',
            chat_id=chat_info['chat_id'],
            chat_name=chat_info['chat_name'],
            sender=sender_name,
            timestamp=message.date,
            metadata=metadata
        )

        return result

    async def sync_all_monitored_chats(
        self,
        limit_per_chat: int = 1000,
        incremental: bool = True
    ) -> Dict:
        """
        Sync all monitored chats

        Args:
            limit_per_chat: Max messages per chat
            incremental: Only sync new messages since last sync

        Returns:
            Overall sync statistics
        """
        overall_stats = {
            'total_chats': len(self.monitored_chats),
            'synced_chats': 0,
            'total_retrieved': 0,
            'total_ingested': 0,
            'total_errors': 0,
            'chat_results': []
        }

        for chat_id, chat_info in self.monitored_chats.items():
            logger.info(f"\nSyncing chat {chat_info['chat_name']}...")

            # Determine offset date for incremental sync
            offset_date = None
            if incremental and chat_info.get('last_sync'):
                try:
                    offset_date = datetime.fromisoformat(chat_info['last_sync'])
                except:
                    pass

            try:
                stats = await self.sync_chat_history(
                    chat_id,
                    limit=limit_per_chat,
                    offset_date=offset_date
                )

                overall_stats['synced_chats'] += 1
                overall_stats['total_retrieved'] += stats['retrieved']
                overall_stats['total_ingested'] += stats['ingested']
                overall_stats['total_errors'] += stats['errors']
                overall_stats['chat_results'].append(stats)

            except Exception as e:
                logger.error(f"Failed to sync chat {chat_info['chat_name']}: {e}")
                overall_stats['total_errors'] += 1

        logger.info(f"\n✓ Overall sync complete:")
        logger.info(f"  Chats synced: {overall_stats['synced_chats']}/{overall_stats['total_chats']}")
        logger.info(f"  Messages ingested: {overall_stats['total_ingested']}")
        logger.info(f"  Errors: {overall_stats['total_errors']}")

        return overall_stats

    async def search_messages(
        self,
        query: str,
        limit: int = 10,
        chat_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search Telegram messages

        Args:
            query: Search query
            limit: Maximum results
            chat_id: Filter by specific chat

        Returns:
            List of matching messages
        """
        filters = {'source': 'telegram'}
        if chat_id:
            filters['chat_id'] = chat_id

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
                'sender': result.document.metadata.get('sender_name', 'Unknown'),
                'timestamp': result.document.timestamp.isoformat(),
                'text': result.document.text,
                'metadata': result.document.metadata
            })

        return messages

    async def get_chat_timeline(
        self,
        chat_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get timeline of messages for a specific chat

        Args:
            chat_id: Chat ID
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
        chat_events = [
            e for e in events
            if e.metadata.get('chat_id') == chat_id and e.metadata.get('source') == 'telegram'
        ]

        return [{
            'timestamp': e.timestamp.isoformat(),
            'sender': e.metadata.get('sender_name', 'Unknown'),
            'text': e.text,
            'metadata': e.metadata
        } for e in chat_events]


async def main():
    """Example usage"""
    print("=== Telegram Integration Test ===\n")

    # Configuration (set your credentials)
    config = TelegramConfig(
        api_id=os.getenv('TELEGRAM_API_ID', ''),
        api_hash=os.getenv('TELEGRAM_API_HASH', ''),
        phone_number=os.getenv('TELEGRAM_PHONE', '')
    )

    if not all([config.api_id, config.api_hash, config.phone_number]):
        print("⚠️  Set TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE environment variables")
        return

    # Initialize
    telegram = TelegramIntegration(config)

    try:
        # Start client
        await telegram.start()

        # Add monitored chat (example)
        # await telegram.add_monitored_chat('@username_or_chat_id')

        # Sync all chats
        # stats = await telegram.sync_all_monitored_chats(limit_per_chat=100)
        # print(f"\nSynced {stats['total_ingested']} messages")

        print("✓ Telegram integration ready")

    finally:
        await telegram.stop()


if __name__ == "__main__":
    asyncio.run(main())
