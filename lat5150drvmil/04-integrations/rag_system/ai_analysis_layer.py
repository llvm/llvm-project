#!/usr/bin/env python3
"""
AI Analysis Layer for Screenshot Intelligence
Advanced event correlation, anomaly detection, and intelligent summarization

Features:
- Automated event linking and pattern discovery
- Anomaly detection in timeline data
- AI-powered summarization using DSMIL AI Engine
- Incident detection and clustering
- Entity extraction and relationship mapping
- Content classification and tagging

Integration:
- Uses DSMIL AI Engine for LLM operations
- Compatible with VectorRAGSystem
- Screenshot Intelligence integration
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# DSMIL AI Engine
try:
    from dsmil_ai_engine import DSMILAIEngine
    AI_ENGINE_AVAILABLE = True
except ImportError:
    AI_ENGINE_AVAILABLE = False
    print("⚠️  DSMIL AI Engine not available")

# Vector RAG
try:
    from vector_rag_system import VectorRAGSystem, Document
    from screenshot_intelligence import ScreenshotIntelligence, Event, Incident
except ImportError:
    VectorRAGSystem = None
    ScreenshotIntelligence = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventLink:
    """Link between two events"""
    source_event_id: str
    target_event_id: str
    link_type: str  # 'temporal', 'content_similar', 'causal', 'concurrent'
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    anomaly_type: str  # 'frequency', 'content', 'pattern', 'temporal'
    event_ids: List[str]
    timestamp: datetime
    severity: float  # 0.0-1.0
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Pattern:
    """Discovered pattern"""
    pattern_id: str
    pattern_type: str  # 'recurring', 'sequence', 'correlation'
    event_ids: List[str]
    frequency: int
    confidence: float
    description: str


class AIAnalysisLayer:
    """
    AI-Powered Analysis Layer

    Provides intelligent event correlation, anomaly detection, and summarization
    """

    def __init__(
        self,
        vector_rag: Optional[VectorRAGSystem] = None,
        screenshot_intel: Optional[ScreenshotIntelligence] = None,
        use_ai_engine: bool = True
    ):
        """
        Initialize AI Analysis Layer

        Args:
            vector_rag: VectorRAGSystem instance
            screenshot_intel: ScreenshotIntelligence instance
            use_ai_engine: Use DSMIL AI Engine for LLM operations
        """
        self.rag = vector_rag if vector_rag else VectorRAGSystem()
        self.intel = screenshot_intel

        # Initialize AI engine
        self.ai_engine = None
        if use_ai_engine and AI_ENGINE_AVAILABLE:
            try:
                self.ai_engine = DSMILAIEngine()
                logger.info("✓ DSMIL AI Engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AI Engine: {e}")

        # Analysis cache
        self.event_links: List[EventLink] = []
        self.anomalies: List[Anomaly] = []
        self.patterns: List[Pattern] = []

        logger.info("✓ AI Analysis Layer initialized")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text (URLs, emails, IPs, etc.)

        Args:
            text: Text to analyze

        Returns:
            Dictionary of entity types and values
        """
        entities = {
            'urls': [],
            'emails': [],
            'ips': [],
            'phone_numbers': [],
            'error_codes': [],
            'file_paths': []
        }

        # URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        entities['urls'] = re.findall(url_pattern, text)

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)

        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        entities['ips'] = re.findall(ip_pattern, text)

        # Phone numbers (basic pattern)
        phone_pattern = r'\+?[\d\s\-\(\)]{10,}'
        entities['phone_numbers'] = re.findall(phone_pattern, text)

        # Error codes (common patterns)
        error_pattern = r'(?:error|err|exception)[\s:]*(?:0x)?[\da-f]+|(?:E|ERR)_[A-Z_\d]+'
        entities['error_codes'] = re.findall(error_pattern, text, re.IGNORECASE)

        # File paths
        path_pattern = r'(?:/[\w.-]+)+/?|(?:[A-Z]:\\(?:[\w.-]+\\)*[\w.-]+)'
        entities['file_paths'] = re.findall(path_pattern, text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def classify_content(self, text: str) -> Dict[str, Any]:
        """
        Classify content type and extract categories

        Args:
            text: Text to classify

        Returns:
            Classification results
        """
        classification = {
            'type': 'unknown',
            'categories': [],
            'confidence': 0.0,
            'keywords': []
        }

        text_lower = text.lower()

        # Content type classification
        if any(keyword in text_lower for keyword in ['error', 'exception', 'failed', 'crash']):
            classification['type'] = 'error_message'
            classification['confidence'] = 0.9
            classification['categories'].append('error')

        elif any(keyword in text_lower for keyword in ['vpn', 'connection', 'network', 'wifi']):
            classification['type'] = 'network_issue'
            classification['confidence'] = 0.8
            classification['categories'].append('network')

        elif any(keyword in text_lower for keyword in ['login', 'authentication', 'password', 'signin']):
            classification['type'] = 'authentication'
            classification['confidence'] = 0.85
            classification['categories'].append('security')

        elif any(keyword in text_lower for keyword in ['database', 'query', 'sql', 'db']):
            classification['type'] = 'database'
            classification['confidence'] = 0.8
            classification['categories'].append('database')

        elif any(keyword in text_lower for keyword in ['api', 'endpoint', 'request', 'response']):
            classification['type'] = 'api_related'
            classification['confidence'] = 0.75
            classification['categories'].append('api')

        # Extract keywords (simple frequency-based)
        words = re.findall(r'\b\w{4,}\b', text_lower)
        word_freq = Counter(words)
        classification['keywords'] = [word for word, _ in word_freq.most_common(10)]

        return classification

    def discover_event_links(
        self,
        events: List[Event],
        time_window: int = 1800,  # 30 minutes
        content_threshold: float = 0.6
    ) -> List[EventLink]:
        """
        Discover links between events

        Args:
            events: List of events to analyze
            time_window: Time window for temporal links (seconds)
            content_threshold: Minimum content similarity

        Returns:
            List of discovered event links
        """
        links = []

        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                # Skip if same event
                if event1.event_id == event2.event_id:
                    continue

                # Calculate time difference
                time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())

                # Temporal link
                if time_diff <= time_window:
                    link_type = 'concurrent' if time_diff < 60 else 'temporal'
                    confidence = 1.0 - (time_diff / time_window)

                    # Extract entities for evidence
                    entities1 = self.extract_entities(event1.content)
                    entities2 = self.extract_entities(event2.content)

                    evidence = []
                    for entity_type in entities1:
                        common = set(entities1[entity_type]) & set(entities2[entity_type])
                        if common:
                            evidence.append(f"Common {entity_type}: {', '.join(common)}")

                    # Content similarity
                    content_words1 = set(event1.content.lower().split())
                    content_words2 = set(event2.content.lower().split())

                    if content_words1 and content_words2:
                        overlap = len(content_words1 & content_words2)
                        union = len(content_words1 | content_words2)
                        similarity = overlap / union if union > 0 else 0.0

                        if similarity >= content_threshold:
                            link_type = 'content_similar'
                            confidence = max(confidence, similarity)
                            evidence.append(f"Content similarity: {similarity:.2%}")

                    if evidence or confidence >= 0.5:
                        link = EventLink(
                            source_event_id=event1.event_id,
                            target_event_id=event2.event_id,
                            link_type=link_type,
                            confidence=confidence,
                            evidence=evidence
                        )
                        links.append(link)

        logger.info(f"✓ Discovered {len(links)} event links")
        return links

    def detect_anomalies(
        self,
        events: List[Event],
        baseline_window: timedelta = timedelta(days=7)
    ) -> List[Anomaly]:
        """
        Detect anomalies in event timeline

        Args:
            events: List of events to analyze
            baseline_window: Time window for baseline

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if not events:
            return anomalies

        # Group events by hour
        hourly_counts = defaultdict(int)
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        # Calculate baseline statistics
        if hourly_counts:
            counts = list(hourly_counts.values())
            avg_count = sum(counts) / len(counts)
            std_dev = (sum((x - avg_count) ** 2 for x in counts) / len(counts)) ** 0.5

            # Detect frequency anomalies (> 2 std deviations)
            for hour, count in hourly_counts.items():
                if count > avg_count + (2 * std_dev):
                    # Find events in this hour
                    hour_events = [
                        e for e in events
                        if e.timestamp.replace(minute=0, second=0, microsecond=0) == hour
                    ]

                    severity = min(1.0, (count - avg_count) / (3 * std_dev))

                    anomaly = Anomaly(
                        anomaly_id=hashlib.md5(f"freq_{hour}".encode()).hexdigest()[:12],
                        anomaly_type='frequency',
                        event_ids=[e.event_id for e in hour_events],
                        timestamp=hour,
                        severity=severity,
                        description=f"Unusual activity spike: {count} events (avg: {avg_count:.1f})",
                        metadata={'count': count, 'avg': avg_count, 'std_dev': std_dev}
                    )
                    anomalies.append(anomaly)

        # Detect content anomalies (unusual error messages, etc.)
        error_events = [e for e in events if 'error' in e.content.lower()]
        if error_events:
            # Group by error type
            error_types = defaultdict(list)
            for event in error_events:
                entities = self.extract_entities(event.content)
                for error_code in entities.get('error_codes', []):
                    error_types[error_code].append(event)

            # Detect rare errors
            for error_code, error_events_list in error_types.items():
                if len(error_events_list) == 1:  # Unique error
                    event = error_events_list[0]
                    anomaly = Anomaly(
                        anomaly_id=hashlib.md5(f"error_{error_code}_{event.event_id}".encode()).hexdigest()[:12],
                        anomaly_type='content',
                        event_ids=[event.event_id],
                        timestamp=event.timestamp,
                        severity=0.7,
                        description=f"Rare error detected: {error_code}",
                        metadata={'error_code': error_code}
                    )
                    anomalies.append(anomaly)

        logger.info(f"✓ Detected {len(anomalies)} anomalies")
        return anomalies

    def discover_patterns(
        self,
        events: List[Event],
        min_frequency: int = 3
    ) -> List[Pattern]:
        """
        Discover recurring patterns in events

        Args:
            events: List of events to analyze
            min_frequency: Minimum pattern frequency

        Returns:
            List of discovered patterns
        """
        patterns = []

        # Pattern 1: Recurring content (similar screenshots/messages)
        content_hashes = defaultdict(list)
        for event in events:
            # Simple content fingerprint
            words = event.content.lower().split()
            if len(words) >= 5:
                fingerprint = ' '.join(sorted(words[:10]))
                content_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:8]
                content_hashes[content_hash].append(event)

        for content_hash, matching_events in content_hashes.items():
            if len(matching_events) >= min_frequency:
                pattern = Pattern(
                    pattern_id=f"recurring_{content_hash}",
                    pattern_type='recurring',
                    event_ids=[e.event_id for e in matching_events],
                    frequency=len(matching_events),
                    confidence=min(1.0, len(matching_events) / (min_frequency * 2)),
                    description=f"Recurring content pattern ({len(matching_events)} occurrences)"
                )
                patterns.append(pattern)

        # Pattern 2: Sequential patterns (A always follows B)
        # Group events by day
        daily_events = defaultdict(list)
        for event in events:
            day_key = event.timestamp.date()
            daily_events[day_key].append(event)

        # Look for sequences
        sequences = defaultdict(int)
        for day, day_events in daily_events.items():
            day_events.sort(key=lambda e: e.timestamp)
            for i in range(len(day_events) - 1):
                event_type1 = day_events[i].event_type
                event_type2 = day_events[i+1].event_type
                sequence_key = f"{event_type1}->{event_type2}"
                sequences[sequence_key] += 1

        for sequence_key, count in sequences.items():
            if count >= min_frequency:
                pattern = Pattern(
                    pattern_id=f"sequence_{hashlib.md5(sequence_key.encode()).hexdigest()[:8]}",
                    pattern_type='sequence',
                    event_ids=[],  # Would need to track specific instances
                    frequency=count,
                    confidence=min(1.0, count / (min_frequency * 2)),
                    description=f"Sequential pattern: {sequence_key} ({count} times)"
                )
                patterns.append(pattern)

        logger.info(f"✓ Discovered {len(patterns)} patterns")
        return patterns

    def generate_incident_summary(
        self,
        incident: Incident,
        max_length: int = 500
    ) -> str:
        """
        Generate AI-powered summary of an incident

        Args:
            incident: Incident to summarize
            max_length: Maximum summary length

        Returns:
            Generated summary
        """
        if not self.ai_engine:
            # Fallback to simple summary
            event_count = len(incident.events)
            duration = (incident.end_time - incident.start_time).total_seconds() / 60
            return (
                f"Incident '{incident.incident_name}' occurred from "
                f"{incident.start_time.strftime('%Y-%m-%d %H:%M')} to "
                f"{incident.end_time.strftime('%H:%M')} ({duration:.0f} minutes). "
                f"Involved {event_count} events across multiple sources."
            )

        # Collect event context
        event_descriptions = []
        for event in incident.events[:10]:  # Limit to 10 events
            time_str = event.timestamp.strftime('%H:%M:%S')
            event_descriptions.append(
                f"[{time_str}] {event.event_type}: {event.content[:100]}"
            )

        context = "\n".join(event_descriptions)

        # Create prompt for AI
        prompt = f"""Analyze this incident and provide a concise summary (max {max_length} characters):

Incident: {incident.incident_name}
Time: {incident.start_time.strftime('%Y-%m-%d %H:%M')} - {incident.end_time.strftime('%H:%M')}
Event Count: {len(incident.events)}

Events:
{context}

Provide a brief summary focusing on:
1. What happened
2. Key events/errors
3. Timeline progression
4. Potential impact

Summary:"""

        try:
            # Use DSMIL AI Engine for summarization
            response = self.ai_engine.query(
                prompt=prompt,
                model="fast",  # Use fast model for summaries
                max_tokens=max_length
            )

            summary = response.get('response', '').strip()
            return summary if summary else "Summary generation failed"

        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            return f"Incident with {len(incident.events)} events from {incident.start_time.strftime('%Y-%m-%d %H:%M')}"

    def analyze_timeline(
        self,
        start_time: datetime,
        end_time: datetime,
        auto_detect_incidents: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive timeline analysis

        Args:
            start_time: Start of timeline
            end_time: End of timeline
            auto_detect_incidents: Automatically detect and group incidents

        Returns:
            Analysis results
        """
        logger.info(f"Analyzing timeline: {start_time} to {end_time}")

        # Get all events in timeline
        events_docs = self.rag.timeline_query(start_time, end_time, limit=10000)

        # Convert to Event objects
        events = []
        for doc in events_docs:
            event = Event(
                event_id=doc.id,
                event_type=doc.doc_type,
                timestamp=doc.timestamp,
                content=doc.text,
                source_device=doc.metadata.get('device_id'),
                source_app=doc.metadata.get('source'),
                metadata=doc.metadata
            )
            events.append(event)

        logger.info(f"Analyzing {len(events)} events")

        # Discover links
        links = self.discover_event_links(events)

        # Detect anomalies
        anomalies = self.detect_anomalies(events)

        # Discover patterns
        patterns = self.discover_patterns(events)

        # Auto-detect incidents
        incidents = []
        if auto_detect_incidents and anomalies:
            # Group anomalies into incidents
            for anomaly in anomalies:
                if anomaly.severity >= 0.7:  # High severity
                    incident_events = [e for e in events if e.event_id in anomaly.event_ids]
                    if incident_events:
                        incident = Incident(
                            incident_id=anomaly.anomaly_id,
                            incident_name=f"Auto-detected: {anomaly.description}",
                            start_time=min(e.timestamp for e in incident_events),
                            end_time=max(e.timestamp for e in incident_events),
                            events=incident_events,
                            tags=['auto-detected', anomaly.anomaly_type]
                        )
                        incidents.append(incident)

        # Generate analysis report
        analysis = {
            'timeline': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'statistics': {
                'total_events': len(events),
                'event_types': dict(Counter(e.event_type for e in events)),
                'sources': dict(Counter(e.metadata.get('source', 'unknown') for e in events))
            },
            'links': {
                'total': len(links),
                'by_type': dict(Counter(l.link_type for l in links)),
                'avg_confidence': sum(l.confidence for l in links) / len(links) if links else 0.0
            },
            'anomalies': {
                'total': len(anomalies),
                'by_type': dict(Counter(a.anomaly_type for a in anomalies)),
                'high_severity': len([a for a in anomalies if a.severity >= 0.7])
            },
            'patterns': {
                'total': len(patterns),
                'by_type': dict(Counter(p.pattern_type for p in patterns))
            },
            'incidents': {
                'total': len(incidents),
                'auto_detected': len(incidents)
            }
        }

        logger.info(f"✓ Analysis complete: {len(links)} links, {len(anomalies)} anomalies, {len(patterns)} patterns")

        return {
            'analysis': analysis,
            'events': events,
            'links': links,
            'anomalies': anomalies,
            'patterns': patterns,
            'incidents': incidents
        }


if __name__ == "__main__":
    print("=== AI Analysis Layer Test ===\n")

    # Initialize
    ai_analysis = AIAnalysisLayer()

    # Test entity extraction
    test_text = "Error connecting to VPN at 192.168.1.1, contact admin@example.com for help. Error code: 0x1234"
    entities = ai_analysis.extract_entities(test_text)
    print("Entities:", json.dumps(entities, indent=2))

    # Test content classification
    classification = ai_analysis.classify_content(test_text)
    print("\nClassification:", json.dumps(classification, indent=2))

    print("\n✓ AI Analysis Layer ready")
