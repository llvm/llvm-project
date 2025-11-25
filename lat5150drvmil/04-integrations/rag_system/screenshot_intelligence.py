#!/usr/bin/env python3
"""
Screenshot Intelligence System
Integrates screenshots, chat logs, and timeline analysis with Vector RAG

Features:
- Screenshot ingestion with OCR
- Chat log correlation (Telegram, Signal)
- Timeline reconstruction
- Event clustering and linking
- Cross-device attribution
- Incident grouping

Integration:
- Uses VectorRAGSystem for storage
- Integrates with existing Telegram scrapers
- Compatible with DSMIL AI Engine
- SWORD Intelligence ready
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import subprocess

# Vector RAG
from vector_rag_system import VectorRAGSystem, Document

# Existing integrations
try:
    from telegram_document_scraper import TelegramDocumentScraper
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information"""
    device_id: str
    device_name: str
    device_type: str  # 'grapheneos', 'laptop', 'pc'
    screenshot_path: Path


@dataclass
class Event:
    """Correlated event"""
    event_id: str
    event_type: str  # 'screenshot', 'chat_message', 'system_log'
    timestamp: datetime
    content: str
    source_device: Optional[str] = None
    source_app: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)


@dataclass
class Incident:
    """Grouped incident"""
    incident_id: str
    incident_name: str
    start_time: datetime
    end_time: datetime
    events: List[Event]
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ScreenshotIntelligence:
    """
    Screenshot Intelligence System

    Manages screenshot ingestion, chat log correlation, and timeline analysis
    """

    def __init__(
        self,
        vector_rag: Optional[VectorRAGSystem] = None,
        data_dir: Path = None
    ):
        """
        Initialize Screenshot Intelligence

        Args:
            vector_rag: VectorRAGSystem instance (creates new if None)
            data_dir: Base directory for screenshots and logs
        """
        # Initialize Vector RAG
        self.rag = vector_rag if vector_rag else VectorRAGSystem()

        # Data directories
        if data_dir is None:
            data_dir = Path.home() / ".screenshot_intel"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.screenshots_dir = self.data_dir / "screenshots"
        self.chat_logs_dir = self.data_dir / "chat_logs"
        self.incidents_dir = self.data_dir / "incidents"

        for d in [self.screenshots_dir, self.chat_logs_dir, self.incidents_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Device registry
        self.devices: Dict[str, DeviceInfo] = {}
        self.load_device_registry()

        # Incidents
        self.incidents: Dict[str, Incident] = {}

        logger.info("✓ Screenshot Intelligence initialized")
        logger.info(f"  Data directory: {self.data_dir}")

    def register_device(
        self,
        device_id: str,
        device_name: str,
        device_type: str,
        screenshot_path: Path
    ):
        """Register a device for screenshot ingestion"""
        self.devices[device_id] = DeviceInfo(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            screenshot_path=Path(screenshot_path)
        )
        self.save_device_registry()
        logger.info(f"✓ Registered device: {device_name} ({device_id})")

    def load_device_registry(self):
        """Load device registry"""
        registry_file = self.data_dir / "devices.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for device_id, device_data in data.items():
                        self.devices[device_id] = DeviceInfo(**device_data)
                logger.info(f"✓ Loaded {len(self.devices)} devices")
            except Exception as e:
                logger.warning(f"Failed to load device registry: {e}")

    def save_device_registry(self):
        """Save device registry"""
        registry_file = self.data_dir / "devices.json"
        data = {
            device_id: {
                'device_id': info.device_id,
                'device_name': info.device_name,
                'device_type': info.device_type,
                'screenshot_path': str(info.screenshot_path)
            }
            for device_id, info in self.devices.items()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def parse_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Parse timestamp from screenshot filename

        Handles formats like:
        - Screenshot_20251111-220341.png (GrapheneOS)
        - screenshot_2025-11-11_22-03-41.png
        - IMG_20251111_220341.jpg
        """
        patterns = [
            (r'(\d{8})[_-](\d{6})', '%Y%m%d%H%M%S'),  # 20251111-220341
            (r'(\d{4})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})[_-](\d{2})', '%Y%m%d%H%M%S'),  # 2025-11-11-22-03-41
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    # Extract all digits
                    timestamp_str = ''.join(match.groups())
                    return datetime.strptime(timestamp_str, fmt)
                except:
                    continue

        return None

    def ingest_screenshot(
        self,
        screenshot_path: Path,
        device_id: Optional[str] = None,
        auto_timestamp: bool = True
    ) -> Dict:
        """
        Ingest a screenshot with OCR

        Args:
            screenshot_path: Path to screenshot
            device_id: Device identifier (optional)
            auto_timestamp: Try to parse timestamp from filename

        Returns:
            Ingestion result
        """
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {'error': f'Screenshot not found: {screenshot_path}'}

        # Parse timestamp
        timestamp = None
        if auto_timestamp:
            timestamp = self.parse_timestamp_from_filename(screenshot_path.name)

        if timestamp is None:
            # Use file modification time
            timestamp = datetime.fromtimestamp(screenshot_path.stat().st_mtime)

        # Prepare metadata
        metadata = {
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': int(timestamp.timestamp()),
            'device_id': device_id,
            'source': 'screenshot'
        }

        if device_id and device_id in self.devices:
            device = self.devices[device_id]
            metadata['device_name'] = device.device_name
            metadata['device_type'] = device.device_type

        # Ingest with Vector RAG (includes OCR)
        result = self.rag.ingest_document(
            screenshot_path,
            doc_type='image',
            metadata=metadata
        )

        if result.get('status') == 'success':
            logger.info(f"✓ Ingested screenshot: {screenshot_path.name}")

        return result

    def scan_device_screenshots(self, device_id: str, pattern: str = "*.png") -> Dict:
        """
        Scan and ingest all screenshots from a device

        Args:
            device_id: Device identifier
            pattern: File pattern (default: *.png)

        Returns:
            Summary of ingestion
        """
        if device_id not in self.devices:
            return {'error': f'Device not registered: {device_id}'}

        device = self.devices[device_id]
        screenshot_dir = device.screenshot_path

        if not screenshot_dir.exists():
            return {'error': f'Screenshot directory not found: {screenshot_dir}'}

        # Find screenshots
        screenshots = list(screenshot_dir.glob(pattern))
        logger.info(f"Found {len(screenshots)} screenshots in {screenshot_dir}")

        results = {
            'device_id': device_id,
            'device_name': device.device_name,
            'total': len(screenshots),
            'success': 0,
            'already_indexed': 0,
            'errors': 0,
            'files': []
        }

        for screenshot in screenshots:
            result = self.ingest_screenshot(screenshot, device_id=device_id)

            if result.get('status') == 'success':
                results['success'] += 1
            elif result.get('status') == 'already_indexed':
                results['already_indexed'] += 1
            else:
                results['errors'] += 1

            results['files'].append({
                'file': screenshot.name,
                'result': result
            })

        logger.info(f"✓ Scan complete: {results['success']} new, {results['already_indexed']} existing, {results['errors']} errors")

        return results

    def find_related_events(
        self,
        event: Event,
        time_window_before: int = 600,  # 10 minutes
        time_window_after: int = 1800,  # 30 minutes
        similarity_threshold: float = 0.6
    ) -> List[Tuple[Event, float, str]]:
        """
        Find events related to a given event

        Args:
            event: Source event
            time_window_before: Time window before event (seconds)
            time_window_after: Time window after event (seconds)
            similarity_threshold: Minimum similarity score

        Returns:
            List of (related_event, score, relation_type)
        """
        related = []

        # Time-based correlation
        start_time = event.timestamp - timedelta(seconds=time_window_before)
        end_time = event.timestamp + timedelta(seconds=time_window_after)

        timeline_events = self.rag.timeline_query(start_time, end_time)

        for doc in timeline_events:
            if doc.id == event.event_id:
                continue  # Skip self

            # Time distance score
            time_diff = abs((doc.timestamp - event.timestamp).total_seconds())
            time_score = 1.0 - (time_diff / (time_window_before + time_window_after))

            # Content similarity (if available)
            content_score = 0.0
            if event.content and doc.text:
                # Simple keyword overlap
                event_words = set(event.content.lower().split())
                doc_words = set(doc.text.lower().split())
                if event_words and doc_words:
                    overlap = len(event_words & doc_words)
                    union = len(event_words | doc_words)
                    content_score = overlap / union if union > 0 else 0.0

            # Combined score
            combined_score = (time_score * 0.4) + (content_score * 0.6)

            if combined_score >= similarity_threshold:
                related_event = Event(
                    event_id=doc.id,
                    event_type=doc.doc_type,
                    timestamp=doc.timestamp,
                    content=doc.text[:500],
                    metadata=doc.metadata
                )

                relation_type = 'temporal'
                if content_score > 0.5:
                    relation_type = 'content_similar'
                if time_diff < 60:
                    relation_type = 'concurrent'

                related.append((related_event, combined_score, relation_type))

        # Sort by score
        related.sort(key=lambda x: x[1], reverse=True)

        return related

    def create_incident(
        self,
        incident_name: str,
        event_ids: List[str],
        tags: Optional[List[str]] = None
    ) -> Incident:
        """
        Create an incident from related events

        Args:
            incident_name: Name for the incident
            event_ids: List of event IDs to include
            tags: Optional tags

        Returns:
            Incident object
        """
        # Fetch events from Qdrant by ID
        events = []
        timestamps = []

        documents = self.rag.get_documents_by_ids(event_ids)

        for doc in documents:
            event = Event(
                event_id=doc.id,
                event_type=doc.doc_type,
                timestamp=doc.timestamp,
                content=doc.text,
                source_device=doc.metadata.get('device_id'),
                source_app=doc.metadata.get('source_app'),
                metadata=doc.metadata
            )
            events.append(event)
            timestamps.append(event.timestamp)

        # Fallback if no documents found
        if not timestamps:
            logger.warning(f"No documents found for event IDs: {event_ids}")
            timestamps = [datetime.now()]

        incident_id = hashlib.md5(incident_name.encode()).hexdigest()[:12]

        incident = Incident(
            incident_id=incident_id,
            incident_name=incident_name,
            start_time=min(timestamps),
            end_time=max(timestamps),
            events=events,
            tags=tags or []
        )

        self.incidents[incident_id] = incident

        # Save incident
        incident_file = self.incidents_dir / f"{incident_id}.json"
        with open(incident_file, 'w') as f:
            json.dump({
                'incident_id': incident.incident_id,
                'incident_name': incident.incident_name,
                'start_time': incident.start_time.isoformat(),
                'end_time': incident.end_time.isoformat(),
                'event_ids': event_ids,
                'event_count': len(events),
                'tags': incident.tags
            }, f, indent=2)

        logger.info(f"✓ Created incident: {incident_name} ({incident_id}) with {len(events)} events")

        return incident

    def generate_timeline_report(
        self,
        start_time: datetime,
        end_time: datetime,
        output_format: str = 'markdown'
    ) -> str:
        """
        Generate a timeline report

        Args:
            start_time: Start of timeline
            end_time: End of timeline
            output_format: Output format ('markdown', 'json')

        Returns:
            Timeline report
        """
        events = self.rag.timeline_query(start_time, end_time, limit=1000)

        if output_format == 'markdown':
            lines = [
                f"# Timeline Report",
                f"",
                f"**Period:** {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total Events:** {len(events)}",
                f"",
                "---",
                ""
            ]

            current_day = None
            for event in events:
                event_day = event.timestamp.strftime('%Y-%m-%d')

                if event_day != current_day:
                    current_day = event_day
                    lines.append(f"\n## {current_day}\n")

                time_str = event.timestamp.strftime('%H:%M:%S')
                event_type = event.doc_type.upper()
                source = event.metadata.get('device_name', event.metadata.get('source', 'Unknown'))

                lines.append(f"**[{time_str}]** `{event_type}` - {source}")
                lines.append(f"  {event.filename if event.filename else event.text[:100]}")
                lines.append("")

            return '\n'.join(lines)

        elif output_format == 'json':
            return json.dumps([{
                'timestamp': e.timestamp.isoformat(),
                'type': e.doc_type,
                'source': e.metadata.get('source', 'unknown'),
                'content': e.text[:200]
            } for e in events], indent=2)

        else:
            return f"Unsupported format: {output_format}"


if __name__ == "__main__":
    print("=== Screenshot Intelligence System Test ===\n")

    # Initialize
    intel = ScreenshotIntelligence()

    # Register devices
    intel.register_device(
        device_id="phone1",
        device_name="GrapheneOS Phone 1",
        device_type="grapheneos",
        screenshot_path=Path.home() / "screenshots" / "phone1"
    )

    intel.register_device(
        device_id="laptop",
        device_name="Dell Latitude 5450",
        device_type="laptop",
        screenshot_path=Path.home() / "screenshots" / "laptop"
    )

    # Get stats
    stats = intel.rag.get_stats()
    print("Vector RAG Stats:", json.dumps(stats, indent=2))

    print("\n✓ Screenshot Intelligence ready")
    print(f"  Devices registered: {len(intel.devices)}")
    print(f"  Data directory: {intel.data_dir}")
