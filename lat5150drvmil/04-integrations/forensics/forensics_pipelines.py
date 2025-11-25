#!/usr/bin/env python3
"""
Forensic Analysis Pipelines

Automated workflows for comprehensive forensic analysis:
1. Evidence Collection Pipeline - Automated capture with forensic metadata
2. Authenticity Verification Pipeline - Batch manipulation detection
3. Device Attribution Pipeline - Device fingerprinting and verification
4. Incident Investigation Pipeline - Comprehensive incident analysis
5. Chain of Custody Pipeline - Hash chain generation and verification
6. Continuous Monitoring Pipeline - Real-time tamper detection

Each pipeline is designed for specific forensic workflows and can be
executed standalone or orchestrated together.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time

from forensics_analyzer import ForensicsAnalyzer
from dbxforensics_toolkit import DBXForensicsToolkit

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a pipeline execution"""
    pipeline_name: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    success: bool
    items_processed: int
    items_success: int
    items_failed: int
    results: Dict
    errors: List[str]
    warnings: List[str]


class ForensicsPipeline:
    """
    Base class for forensic analysis pipelines

    Provides common functionality for all pipeline types:
    - Progress tracking
    - Error handling
    - Result aggregation
    - Logging and reporting
    """

    def __init__(
        self,
        name: str,
        toolkit: Optional[DBXForensicsToolkit] = None,
        analyzer: Optional[ForensicsAnalyzer] = None
    ):
        """
        Initialize pipeline

        Args:
            name: Pipeline name
            toolkit: DBXForensicsToolkit instance
            analyzer: ForensicsAnalyzer instance
        """
        self.name = name
        self.toolkit = toolkit or DBXForensicsToolkit()
        self.analyzer = analyzer or ForensicsAnalyzer(toolkit=self.toolkit)

        self.execution_id = None
        self.start_time = None
        self.end_time = None

        logger.info(f"✓ Pipeline initialized: {name}")

    def execute(self, *args, **kwargs) -> PipelineResult:
        """Execute pipeline (override in subclass)"""
        raise NotImplementedError("Subclass must implement execute()")

    def _start_execution(self):
        """Start pipeline execution"""
        self.execution_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        logger.info(f"▶️  Starting pipeline: {self.name} ({self.execution_id})")

    def _end_execution(self, success: bool = True):
        """End pipeline execution"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        status = "✓ COMPLETED" if success else "✗ FAILED"
        logger.info(f"{status}: {self.name} (duration: {duration:.1f}s)")


class EvidenceCollectionPipeline(ForensicsPipeline):
    """
    Automated Evidence Collection Pipeline

    Workflow:
    1. Capture screenshots with forensic metadata (dbxScreenshot)
    2. Calculate cryptographic hashes (dbxHashFile)
    3. Extract comprehensive metadata (dbxMetadata)
    4. Generate chain of custody entry
    5. Ingest into intelligence database

    Use Cases:
    - Scheduled evidence capture
    - Automated monitoring
    - Incident documentation
    """

    def __init__(self, toolkit=None, analyzer=None, output_dir: Path = None):
        super().__init__("Evidence Collection", toolkit, analyzer)

        self.output_dir = output_dir or Path.cwd() / "forensic_evidence"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        capture_count: int = 1,
        capture_interval: int = 0,
        region: Optional[tuple] = None
    ) -> PipelineResult:
        """
        Execute evidence collection

        Args:
            capture_count: Number of screenshots to capture
            capture_interval: Seconds between captures (0 = single capture)
            region: Screen region to capture (x, y, width, height)

        Returns:
            PipelineResult with collection summary
        """
        self._start_execution()

        results = {
            'captures': [],
            'chain_of_custody': []
        }

        errors = []
        warnings = []

        items_success = 0
        items_failed = 0

        for i in range(capture_count):
            try:
                # 1. Capture screenshot
                timestamp = datetime.now()
                screenshot_name = f"forensic_capture_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i+1}.png"
                screenshot_path = self.output_dir / screenshot_name

                capture_result = self.toolkit.screenshot.capture(
                    output_path=screenshot_path,
                    region=region
                )

                if not capture_result.success:
                    errors.append(f"Capture {i+1} failed: {capture_result.stderr}")
                    items_failed += 1
                    continue

                # 2. Calculate hashes
                hash_result = self.toolkit.hash_file.calculate_hashes(
                    file_path=screenshot_path,
                    algorithms=['sha256', 'sha512', 'sha3-256']
                )

                hashes = {}
                if hash_result.success:
                    hashes = self.toolkit.hash_file.parse_output(
                        hash_result.stdout,
                        hash_result.stderr
                    )

                # 3. Extract metadata
                metadata_result = self.toolkit.metadata.extract(
                    file_path=screenshot_path,
                    output_format='json'
                )

                metadata = {}
                if metadata_result.success:
                    metadata = self.toolkit.metadata.parse_output(
                        metadata_result.stdout,
                        metadata_result.stderr
                    )

                # 4. Chain of custody entry
                custody_entry = {
                    'capture_id': i + 1,
                    'timestamp': timestamp.isoformat(),
                    'screenshot_path': str(screenshot_path),
                    'sha256': hashes.get('sha256', ''),
                    'sha512': hashes.get('sha512', ''),
                    'sha3_256': hashes.get('sha3-256', ''),
                    'metadata': metadata,
                    'captured_by': 'LAT5150_ForensicsPipeline',
                    'pipeline_execution': self.execution_id
                }

                results['captures'].append({
                    'screenshot': screenshot_name,
                    'timestamp': timestamp.isoformat(),
                    'hashes': hashes
                })

                results['chain_of_custody'].append(custody_entry)

                items_success += 1

                logger.info(f"✓ Capture {i+1}/{capture_count}: {screenshot_name}")

                # Wait before next capture
                if i < capture_count - 1 and capture_interval > 0:
                    time.sleep(capture_interval)

            except Exception as e:
                error_msg = f"Capture {i+1} exception: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                items_failed += 1

        # Save chain of custody
        custody_file = self.output_dir / f"chain_of_custody_{self.execution_id}.json"
        with open(custody_file, 'w') as f:
            json.dump(results['chain_of_custody'], f, indent=2)

        logger.info(f"✓ Chain of custody saved: {custody_file}")

        self._end_execution(success=(items_failed == 0))

        return PipelineResult(
            pipeline_name=self.name,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            success=(items_failed == 0),
            items_processed=capture_count,
            items_success=items_success,
            items_failed=items_failed,
            results=results,
            errors=errors,
            warnings=warnings
        )


class AuthenticityVerificationPipeline(ForensicsPipeline):
    """
    Batch Authenticity Verification Pipeline

    Workflow:
    1. Load batch of screenshots
    2. ELA analysis for each (dbxELA)
    3. Noise map analysis for device fingerprinting (dbxNoiseMap)
    4. Aggregate authenticity scores
    5. Generate verification report

    Use Cases:
    - Verify evidence collection integrity
    - Detect tampered screenshots
    - Batch processing of historical evidence
    """

    def execute(
        self,
        screenshot_paths: List[Path],
        quality_threshold: int = 90
    ) -> PipelineResult:
        """
        Execute batch authenticity verification

        Args:
            screenshot_paths: List of screenshot paths to verify
            quality_threshold: JPEG quality for ELA (default: 90)

        Returns:
            PipelineResult with verification summary
        """
        self._start_execution()

        results = {
            'authentic': [],
            'suspicious': [],
            'tampered': [],
            'analysis_details': []
        }

        errors = []
        warnings = []

        items_success = 0
        items_failed = 0

        for screenshot in screenshot_paths:
            try:
                # Run comprehensive analysis
                analysis = self.analyzer.analyze_screenshot(
                    image_path=screenshot,
                    expected_device_id=None
                )

                # Categorize by verdict
                verdict = analysis.forensic_verdict

                details = {
                    'screenshot': screenshot.name,
                    'verdict': verdict,
                    'authenticity_score': analysis.authenticity_score,
                    'manipulation_detected': analysis.manipulation_detected,
                    'confidence': analysis.confidence_score,
                    'flags': analysis.flags,
                    'warnings': analysis.warnings
                }

                results['analysis_details'].append(details)

                if verdict == 'authentic':
                    results['authentic'].append(screenshot.name)
                elif verdict == 'suspicious':
                    results['suspicious'].append(screenshot.name)
                    warnings.append(f"Suspicious: {screenshot.name}")
                elif verdict == 'tampered':
                    results['tampered'].append(screenshot.name)
                    warnings.append(f"TAMPERED: {screenshot.name}")

                items_success += 1

                logger.info(f"✓ Verified {screenshot.name}: {verdict.upper()}")

            except Exception as e:
                error_msg = f"Failed to verify {screenshot.name}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                items_failed += 1

        # Calculate statistics
        total = len(screenshot_paths)
        results['statistics'] = {
            'total': total,
            'authentic': len(results['authentic']),
            'suspicious': len(results['suspicious']),
            'tampered': len(results['tampered']),
            'authenticity_rate': (len(results['authentic']) / total * 100) if total > 0 else 0,
            'tamper_rate': (len(results['tampered']) / total * 100) if total > 0 else 0
        }

        self._end_execution(success=(items_failed == 0))

        return PipelineResult(
            pipeline_name=self.name,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            success=(items_failed == 0),
            items_processed=total,
            items_success=items_success,
            items_failed=items_failed,
            results=results,
            errors=errors,
            warnings=warnings
        )


class IncidentInvestigationPipeline(ForensicsPipeline):
    """
    Comprehensive Incident Investigation Pipeline

    Workflow:
    1. Load incident screenshots
    2. Verify integrity (hashes)
    3. Check sequence completeness (dbxSeqCheck)
    4. Batch authenticity verification
    5. Timeline reconstruction
    6. Generate investigation report

    Use Cases:
    - Post-incident analysis
    - Evidence validation
    - Court-ready documentation
    """

    def execute(
        self,
        screenshot_paths: List[Path],
        incident_name: str,
        expected_hashes: Optional[Dict[str, str]] = None
    ) -> PipelineResult:
        """
        Execute incident investigation

        Args:
            screenshot_paths: List of screenshot paths in incident
            incident_name: Name of incident
            expected_hashes: Optional dict of filename -> sha256 hash for verification

        Returns:
            PipelineResult with investigation summary
        """
        self._start_execution()

        results = {
            'incident_name': incident_name,
            'total_screenshots': len(screenshot_paths),
            'integrity_check': {},
            'sequence_check': {},
            'authenticity_check': {},
            'timeline': []
        }

        errors = []
        warnings = []

        # 1. Integrity verification
        logger.info("Step 1: Integrity verification...")

        integrity_verified = 0
        integrity_failed = 0

        if expected_hashes:
            for screenshot in screenshot_paths:
                try:
                    # Calculate current hash
                    hash_result = self.toolkit.hash_file.calculate_hashes(
                        file_path=screenshot,
                        algorithms=['sha256']
                    )

                    if hash_result.success:
                        parsed = self.toolkit.hash_file.parse_output(
                            hash_result.stdout,
                            hash_result.stderr
                        )

                        current_hash = parsed.get('sha256', '')
                        expected_hash = expected_hashes.get(screenshot.name, '')

                        if current_hash == expected_hash:
                            integrity_verified += 1
                        else:
                            integrity_failed += 1
                            warnings.append(f"Hash mismatch: {screenshot.name}")

                except Exception as e:
                    errors.append(f"Hash verification failed for {screenshot.name}: {e}")

            results['integrity_check'] = {
                'verified': integrity_verified,
                'failed': integrity_failed,
                'total': len(expected_hashes)
            }

        # 2. Sequence completeness check
        logger.info("Step 2: Sequence verification...")

        try:
            # Extract timestamps from filenames
            timestamps = []

            for screenshot in screenshot_paths:
                # Parse timestamp from filename (assuming format with unix timestamp or structured date)
                # This is a simplified example - adjust based on actual filename format
                import re
                match = re.search(r'(\d{10,})', screenshot.name)
                if match:
                    timestamps.append(int(match.group(1)))

            if timestamps:
                seq_result = self.toolkit.seq_check.check_sequence(sorted(timestamps))

                if seq_result.success:
                    parsed = self.toolkit.seq_check.parse_output(
                        seq_result.stdout,
                        seq_result.stderr
                    )
                    results['sequence_check'] = parsed

                    if not parsed.get('complete', True):
                        warnings.append(f"Sequence gaps detected: {parsed.get('missing', [])}")

        except Exception as e:
            errors.append(f"Sequence check failed: {e}")

        # 3. Batch authenticity verification
        logger.info("Step 3: Authenticity verification...")

        batch_report = self.analyzer.batch_analyze(
            image_paths=screenshot_paths,
            check_sequence=False  # Already checked above
        )

        results['authenticity_check'] = {
            'authentic': batch_report.authentic_count,
            'suspicious': batch_report.suspicious_count,
            'tampered': batch_report.tampered_count,
            'authenticity_rate': batch_report.authenticity_rate,
            'tampered_files': [r.screenshot_path.name for r in batch_report.results if r.forensic_verdict == 'tampered']
        }

        # 4. Timeline reconstruction
        logger.info("Step 4: Timeline reconstruction...")

        for screenshot in sorted(screenshot_paths, key=lambda p: p.stat().st_mtime):
            timeline_entry = {
                'timestamp': datetime.fromtimestamp(screenshot.stat().st_mtime).isoformat(),
                'screenshot': screenshot.name,
                'size': screenshot.stat().st_size
            }
            results['timeline'].append(timeline_entry)

        # Overall verdict
        all_verified = (
            integrity_failed == 0 and
            results['sequence_check'].get('complete', False) and
            results['authenticity_check']['tampered'] == 0
        )

        results['investigation_verdict'] = 'VERIFIED' if all_verified else 'COMPROMISED'

        self._end_execution(success=all_verified)

        return PipelineResult(
            pipeline_name=self.name,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            success=all_verified,
            items_processed=len(screenshot_paths),
            items_success=len(screenshot_paths) if all_verified else 0,
            items_failed=0 if all_verified else len(screenshot_paths),
            results=results,
            errors=errors,
            warnings=warnings
        )


class ContinuousMonitoringPipeline(ForensicsPipeline):
    """
    Continuous Monitoring Pipeline

    Workflow:
    1. Watch directory for new screenshots
    2. Auto-ingest with forensic analysis
    3. Real-time tamper detection
    4. Alert generation
    5. Continuous hash chain update

    Use Cases:
    - Real-time evidence collection
    - Automated monitoring
    - Instant tamper alerts
    """

    def __init__(self, toolkit=None, analyzer=None, alert_callback: Optional[Callable] = None):
        super().__init__("Continuous Monitoring", toolkit, analyzer)
        self.alert_callback = alert_callback

    def execute(
        self,
        watch_directory: Path,
        duration_seconds: int = 60,
        check_interval: int = 5
    ) -> PipelineResult:
        """
        Execute continuous monitoring

        Args:
            watch_directory: Directory to monitor
            duration_seconds: How long to monitor (seconds)
            check_interval: Seconds between checks

        Returns:
            PipelineResult with monitoring summary
        """
        self._start_execution()

        watch_directory = Path(watch_directory)

        if not watch_directory.exists():
            watch_directory.mkdir(parents=True, exist_ok=True)

        results = {
            'monitored_directory': str(watch_directory),
            'duration': duration_seconds,
            'files_detected': [],
            'alerts': []
        }

        errors = []
        warnings = []

        # Track processed files
        processed = set()

        # Initial scan
        for screenshot in watch_directory.glob("*.png"):
            processed.add(screenshot.name)

        logger.info(f"Monitoring {watch_directory} for {duration_seconds}s...")

        start = time.time()
        items_success = 0
        items_failed = 0

        while (time.time() - start) < duration_seconds:
            try:
                # Find new files
                current_files = set(f.name for f in watch_directory.glob("*.png"))
                new_files = current_files - processed

                for filename in new_files:
                    filepath = watch_directory / filename

                    try:
                        # Analyze new file
                        analysis = self.analyzer.analyze_screenshot(filepath)

                        results['files_detected'].append({
                            'filename': filename,
                            'detected_at': datetime.now().isoformat(),
                            'verdict': analysis.forensic_verdict,
                            'authenticity_score': analysis.authenticity_score
                        })

                        # Alert on tampering
                        if analysis.manipulation_detected:
                            alert = {
                                'timestamp': datetime.now().isoformat(),
                                'filename': filename,
                                'verdict': analysis.forensic_verdict,
                                'flags': analysis.flags
                            }

                            results['alerts'].append(alert)
                            warnings.append(f"ALERT: Tampered file detected: {filename}")

                            if self.alert_callback:
                                self.alert_callback(alert)

                        processed.add(filename)
                        items_success += 1

                        logger.info(f"✓ Processed new file: {filename} - {analysis.forensic_verdict}")

                    except Exception as e:
                        errors.append(f"Failed to process {filename}: {e}")
                        items_failed += 1

                # Wait before next check
                time.sleep(check_interval)

            except Exception as e:
                errors.append(f"Monitoring error: {e}")

        self._end_execution(success=(len(errors) == 0))

        return PipelineResult(
            pipeline_name=self.name,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            success=(len(errors) == 0),
            items_processed=len(results['files_detected']),
            items_success=items_success,
            items_failed=items_failed,
            results=results,
            errors=errors,
            warnings=warnings
        )


if __name__ == "__main__":
    print("=== Forensic Analysis Pipelines Test ===\n")

    # Test Evidence Collection Pipeline
    print("1. Evidence Collection Pipeline")
    print("-" * 50)

    collection = EvidenceCollectionPipeline()
    result = collection.execute(capture_count=3, capture_interval=2)

    print(f"\nExecution ID: {result.execution_id}")
    print(f"Success: {result.success}")
    print(f"Items processed: {result.items_processed}")
    print(f"Chain of custody entries: {len(result.results['chain_of_custody'])}")

    print("\n✓ Pipeline tests complete")
