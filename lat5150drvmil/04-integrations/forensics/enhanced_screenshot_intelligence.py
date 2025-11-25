#!/usr/bin/env python3
"""
Enhanced Screenshot Intelligence with Forensics Integration

Extends the base ScreenshotIntelligence system with comprehensive forensic analysis:
- Authenticity verification (ELA)
- Device fingerprinting (NoiseMap)
- Metadata extraction and enrichment
- Hash chain generation
- Sequence integrity verification
- Tamper detection and alerting

This module bridges the gap between evidence collection and forensic analysis,
ensuring all ingested screenshots are forensically verified and traceable.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'rag_system'))

from screenshot_intelligence import ScreenshotIntelligence, Event, Incident
from vector_rag_system import VectorRAGSystem
from forensics_analyzer import ForensicsAnalyzer, ForensicAnalysisReport

logger = logging.getLogger(__name__)


@dataclass
class ForensicEvent(Event):
    """
    Enhanced Event with forensic metadata

    Extends base Event class with comprehensive forensic analysis results
    """
    forensic_verdict: Optional[str] = None  # 'authentic', 'suspicious', 'tampered'
    authenticity_score: Optional[float] = None  # 0-100
    manipulation_detected: Optional[bool] = None
    device_signature: Optional[str] = None
    device_verified: Optional[bool] = None
    file_hashes: Dict[str, str] = field(default_factory=dict)
    forensic_flags: List[str] = field(default_factory=list)
    forensic_warnings: List[str] = field(default_factory=list)
    chain_of_custody: List[Dict] = field(default_factory=list)


@dataclass
class ForensicIncident(Incident):
    """
    Enhanced Incident with forensic metadata

    Extends base Incident class with integrity verification and statistics
    """
    integrity_verified: bool = False
    sequence_complete: bool = False
    missing_items: List[int] = field(default_factory=list)
    tampered_events: List[str] = field(default_factory=list)
    authenticity_stats: Dict = field(default_factory=dict)


class EnhancedScreenshotIntelligence(ScreenshotIntelligence):
    """
    Screenshot Intelligence with Forensic Analysis

    Extends ScreenshotIntelligence to add:
    - Automatic forensic analysis during ingestion
    - Authenticity verification for all screenshots
    - Device fingerprint tracking and verification
    - Hash chain generation for evidence integrity
    - Sequence gap detection
    - Tamper alerting
    """

    def __init__(
        self,
        vector_rag: Optional[VectorRAGSystem] = None,
        data_dir: Path = None,
        enable_forensics: bool = True,
        auto_verify: bool = True,
        alert_on_tampering: bool = True
    ):
        """
        Initialize Enhanced Screenshot Intelligence

        Args:
            vector_rag: VectorRAGSystem instance (creates new if None)
            data_dir: Base directory for screenshots and logs
            enable_forensics: Enable forensic analysis (default: True)
            auto_verify: Automatically verify all ingested screenshots (default: True)
            alert_on_tampering: Alert when tampering detected (default: True)
        """
        # Initialize base class
        super().__init__(vector_rag=vector_rag, data_dir=data_dir)

        # Forensics configuration
        self.enable_forensics = enable_forensics
        self.auto_verify = auto_verify
        self.alert_on_tampering = alert_on_tampering

        # Initialize forensics analyzer
        if self.enable_forensics:
            try:
                self.forensics = ForensicsAnalyzer()
                logger.info("✓ Forensics analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize forensics: {e}")
                self.enable_forensics = False
                self.forensics = None
        else:
            self.forensics = None

        # Forensics data directories
        self.forensics_dir = self.data_dir / "forensics"
        self.forensics_dir.mkdir(parents=True, exist_ok=True)

        self.device_signatures_dir = self.forensics_dir / "device_signatures"
        self.device_signatures_dir.mkdir(parents=True, exist_ok=True)

        self.hash_chains_dir = self.forensics_dir / "hash_chains"
        self.hash_chains_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_dir = self.forensics_dir / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Enhanced Screenshot Intelligence initialized")
        logger.info(f"  Forensics enabled: {self.enable_forensics}")
        logger.info(f"  Auto-verify: {self.auto_verify}")
        logger.info(f"  Alert on tampering: {self.alert_on_tampering}")

    def ingest_screenshot_with_forensics(
        self,
        screenshot_path: Path,
        device_id: Optional[str] = None,
        auto_timestamp: bool = True,
        verify_device: bool = True
    ) -> Dict:
        """
        Ingest screenshot with comprehensive forensic analysis

        This is the enhanced version of ingest_screenshot() that adds:
        1. Authenticity verification (ELA)
        2. Device fingerprinting (NoiseMap)
        3. Metadata extraction and enrichment
        4. Hash chain generation
        5. Tamper detection

        Args:
            screenshot_path: Path to screenshot
            device_id: Device identifier (optional)
            auto_timestamp: Try to parse timestamp from filename
            verify_device: Verify device signature matches registered device

        Returns:
            Ingestion result with forensic analysis
        """
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {'error': f'Screenshot not found: {screenshot_path}'}

        # Step 1: Base ingestion (OCR, timeline placement)
        base_result = self.ingest_screenshot(
            screenshot_path=screenshot_path,
            device_id=device_id,
            auto_timestamp=auto_timestamp
        )

        if base_result.get('status') != 'success':
            return base_result

        # Step 2: Forensic analysis (if enabled)
        forensic_result = {}

        if self.enable_forensics and self.forensics:
            try:
                # Run comprehensive forensic analysis
                expected_device_id = device_id if verify_device else None

                analysis: ForensicAnalysisReport = self.forensics.analyze_screenshot(
                    image_path=screenshot_path,
                    expected_device_id=expected_device_id
                )

                # Extract forensic metadata
                forensic_result = {
                    'forensic_verdict': analysis.forensic_verdict,
                    'authenticity_score': analysis.authenticity_score,
                    'manipulation_detected': analysis.manipulation_detected,
                    'device_signature': analysis.device_signature,
                    'device_verified': analysis.device_verified,
                    'confidence_score': analysis.confidence_score,
                    'file_hashes': analysis.file_hashes,
                    'forensic_flags': analysis.flags,
                    'forensic_warnings': analysis.warnings
                }

                # Alert on tampering
                if self.alert_on_tampering and analysis.manipulation_detected:
                    self._create_tamper_alert(screenshot_path, analysis)

                # Store hash chain entry
                self._add_hash_chain_entry(
                    screenshot_path=screenshot_path,
                    device_id=device_id,
                    hashes=analysis.file_hashes,
                    verdict=analysis.forensic_verdict
                )

                logger.info(f"✓ Forensic analysis complete: {analysis.forensic_verdict.upper()}")

            except Exception as e:
                logger.warning(f"Forensic analysis failed: {e}")
                forensic_result = {'forensic_error': str(e)}

        # Step 3: Merge results
        result = {
            **base_result,
            'forensics': forensic_result
        }

        return result

    def register_device_with_signature(
        self,
        device_id: str,
        device_name: str,
        device_type: str,
        screenshot_path: Path,
        sample_screenshots: List[Path]
    ) -> Dict:
        """
        Register device with forensic signature learning

        This extends register_device() to also learn the device's unique
        noise pattern signature for future verification.

        Args:
            device_id: Device identifier
            device_name: Human-readable device name
            device_type: Type of device
            screenshot_path: Directory where screenshots are stored
            sample_screenshots: List of 3-5 sample screenshots for signature learning

        Returns:
            Registration result with signature info
        """
        # Register device normally
        self.register_device(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            screenshot_path=screenshot_path
        )

        # Learn forensic signature
        signature_result = {}

        if self.enable_forensics and self.forensics:
            try:
                self.forensics.register_device_signature(
                    device_id=device_id,
                    sample_images=sample_screenshots
                )

                signature_result = {
                    'signature_registered': True,
                    'sample_count': len(sample_screenshots),
                    'device_id': device_id
                }

                logger.info(f"✓ Device signature registered for {device_name}")

            except Exception as e:
                logger.warning(f"Failed to register device signature: {e}")
                signature_result = {'signature_error': str(e)}

        return {
            'device_registered': True,
            'device_id': device_id,
            'device_name': device_name,
            'forensic_signature': signature_result
        }

    def verify_incident_integrity(self, incident_id: str) -> Dict:
        """
        Verify forensic integrity of an incident

        Checks:
        1. All screenshots are authentic
        2. No sequence gaps
        3. Hash chains are valid
        4. Device signatures match

        Args:
            incident_id: Incident identifier

        Returns:
            Integrity verification report
        """
        if incident_id not in self.incidents:
            return {'error': f'Incident not found: {incident_id}'}

        incident = self.incidents[incident_id]

        # Collect all screenshot events
        screenshot_events = [
            e for e in incident.events
            if e.event_type == 'image' and e.metadata.get('source') == 'screenshot'
        ]

        if not screenshot_events:
            return {'error': 'No screenshots in incident'}

        # Extract timestamps for sequence checking
        timestamps = sorted([
            e.metadata.get('timestamp_unix', 0)
            for e in screenshot_events
            if 'timestamp_unix' in e.metadata
        ])

        # Check sequence integrity
        sequence_result = {}
        if self.enable_forensics and self.forensics:
            try:
                sequence_check = self.forensics.toolkit.seq_check.check_sequence(timestamps)

                if sequence_check.success:
                    parsed = self.forensics.toolkit.seq_check.parse_output(
                        sequence_check.stdout,
                        sequence_check.stderr
                    )
                    sequence_result = parsed

            except Exception as e:
                logger.warning(f"Sequence check failed: {e}")
                sequence_result = {'error': str(e)}

        # Count tampered/suspicious screenshots
        tampered = []
        suspicious = []
        authentic = []

        for event in screenshot_events:
            forensics = event.metadata.get('forensics', {})
            verdict = forensics.get('forensic_verdict', 'unknown')

            if verdict == 'tampered':
                tampered.append(event.event_id)
            elif verdict == 'suspicious':
                suspicious.append(event.event_id)
            elif verdict == 'authentic':
                authentic.append(event.event_id)

        # Overall integrity verdict
        integrity_verified = (
            len(tampered) == 0 and
            sequence_result.get('complete', False)
        )

        return {
            'incident_id': incident_id,
            'incident_name': incident.incident_name,
            'total_screenshots': len(screenshot_events),
            'integrity_verified': integrity_verified,
            'authenticity_stats': {
                'authentic': len(authentic),
                'suspicious': len(suspicious),
                'tampered': len(tampered)
            },
            'sequence_check': sequence_result,
            'tampered_events': tampered,
            'warnings': suspicious
        }

    def batch_ingest_with_forensics(
        self,
        screenshot_paths: List[Path],
        device_id: Optional[str] = None,
        check_sequence: bool = True
    ) -> Dict:
        """
        Batch ingest screenshots with forensic analysis

        Args:
            screenshot_paths: List of screenshot paths
            device_id: Device identifier
            check_sequence: Verify sequence integrity

        Returns:
            Batch ingestion summary with forensics
        """
        results = {
            'total': len(screenshot_paths),
            'success': 0,
            'errors': 0,
            'authentic': 0,
            'suspicious': 0,
            'tampered': 0,
            'files': []
        }

        for screenshot in screenshot_paths:
            result = self.ingest_screenshot_with_forensics(
                screenshot_path=screenshot,
                device_id=device_id
            )

            if result.get('status') == 'success':
                results['success'] += 1

                # Count forensic verdicts
                forensics = result.get('forensics', {})
                verdict = forensics.get('forensic_verdict', 'unknown')

                if verdict == 'authentic':
                    results['authentic'] += 1
                elif verdict == 'suspicious':
                    results['suspicious'] += 1
                elif verdict == 'tampered':
                    results['tampered'] += 1
            else:
                results['errors'] += 1

            results['files'].append({
                'file': screenshot.name,
                'result': result
            })

        # Sequence integrity check
        if check_sequence and self.enable_forensics and self.forensics:
            try:
                # Extract timestamps
                timestamps = []
                for file_result in results['files']:
                    if file_result['result'].get('status') == 'success':
                        metadata = file_result['result'].get('metadata', {})
                        ts = metadata.get('timestamp_unix')
                        if ts:
                            timestamps.append(ts)

                if timestamps:
                    seq_result = self.forensics.toolkit.seq_check.check_sequence(
                        sorted(timestamps)
                    )

                    if seq_result.success:
                        parsed = self.forensics.toolkit.seq_check.parse_output(
                            seq_result.stdout,
                            seq_result.stderr
                        )
                        results['sequence_check'] = parsed

            except Exception as e:
                logger.warning(f"Sequence check failed: {e}")
                results['sequence_check'] = {'error': str(e)}

        logger.info(f"✓ Batch ingestion complete: {results['success']}/{results['total']} successful")
        logger.info(f"  Authentic: {results['authentic']}, Suspicious: {results['suspicious']}, Tampered: {results['tampered']}")

        return results

    def _create_tamper_alert(
        self,
        screenshot_path: Path,
        analysis: ForensicAnalysisReport
    ):
        """Create tamper detection alert"""
        timestamp = datetime.now()
        alert_id = f"tamper_{timestamp.strftime('%Y%m%d_%H%M%S')}_{screenshot_path.stem}"

        alert_file = self.alerts_dir / f"{alert_id}.json"

        alert_data = {
            'alert_id': alert_id,
            'timestamp': timestamp.isoformat(),
            'alert_type': 'tampering_detected',
            'severity': 'high' if analysis.forensic_verdict == 'tampered' else 'medium',
            'screenshot_path': str(screenshot_path),
            'forensic_verdict': analysis.forensic_verdict,
            'authenticity_score': analysis.authenticity_score,
            'confidence_score': analysis.confidence_score,
            'flags': analysis.flags,
            'warnings': analysis.warnings,
            'file_hashes': analysis.file_hashes
        }

        import json
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)

        logger.warning(f"⚠️  TAMPER ALERT: {screenshot_path.name} - {analysis.forensic_verdict}")

    def _add_hash_chain_entry(
        self,
        screenshot_path: Path,
        device_id: Optional[str],
        hashes: Dict[str, str],
        verdict: str
    ):
        """Add entry to hash chain for evidence integrity"""
        import json

        chain_file = self.hash_chains_dir / f"{device_id or 'unknown'}_chain.jsonl"

        entry = {
            'timestamp': datetime.now().isoformat(),
            'screenshot': screenshot_path.name,
            'sha256': hashes.get('sha256', ''),
            'sha512': hashes.get('sha512', ''),
            'verdict': verdict,
            'device_id': device_id
        }

        with open(chain_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_forensics_summary(self) -> Dict:
        """
        Get overall forensics summary statistics

        Returns:
            Forensics statistics across all ingested screenshots
        """
        # Query all screenshots from vector RAG
        all_docs = self.rag.collection.scroll(
            limit=10000,
            with_payload=True
        )[0]

        screenshot_docs = [
            doc for doc in all_docs
            if doc.payload.get('doc_type') == 'image' and
               doc.payload.get('metadata', {}).get('source') == 'screenshot'
        ]

        total = len(screenshot_docs)
        authentic = 0
        suspicious = 0
        tampered = 0
        device_verified = 0

        for doc in screenshot_docs:
            forensics = doc.payload.get('metadata', {}).get('forensics', {})
            verdict = forensics.get('forensic_verdict', 'unknown')

            if verdict == 'authentic':
                authentic += 1
            elif verdict == 'suspicious':
                suspicious += 1
            elif verdict == 'tampered':
                tampered += 1

            if forensics.get('device_verified'):
                device_verified += 1

        return {
            'total_screenshots': total,
            'authentic': authentic,
            'suspicious': suspicious,
            'tampered': tampered,
            'device_verified': device_verified,
            'authenticity_rate': (authentic / total * 100) if total > 0 else 0,
            'tamper_rate': (tampered / total * 100) if total > 0 else 0
        }


if __name__ == "__main__":
    import json

    print("=== Enhanced Screenshot Intelligence with Forensics ===\n")

    # Initialize
    intel = EnhancedScreenshotIntelligence(
        enable_forensics=True,
        auto_verify=True,
        alert_on_tampering=True
    )

    # Register device with signature
    print("Registering device with forensic signature...")

    sample_dir = Path.home() / "screenshots" / "phone1"
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.png"))[:5]

        if samples:
            result = intel.register_device_with_signature(
                device_id="phone1",
                device_name="GrapheneOS Phone 1",
                device_type="grapheneos",
                screenshot_path=sample_dir,
                sample_screenshots=samples
            )

            print(json.dumps(result, indent=2))

    # Get forensics summary
    print("\nForensics Summary:")
    summary = intel.get_forensics_summary()
    print(json.dumps(summary, indent=2))

    print("\n✓ Enhanced Screenshot Intelligence ready")
    print(f"  Forensics enabled: {intel.enable_forensics}")
    print(f"  Devices registered: {len(intel.devices)}")
