#!/usr/bin/env python3
"""
Forensics Analyzer - High-Level Orchestrator
Coordinates all 9 DBXForensics tools for comprehensive analysis workflows

Automated Pipelines:
- Full screenshot forensic analysis
- Batch evidence processing
- Incident investigation
- Evidence integrity verification
- Device fingerprinting
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from dbxforensics_toolkit import DBXForensicsToolkit, ToolResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForensicAnalysisReport:
    """Comprehensive forensic analysis report"""
    file_path: str
    analysis_timestamp: str

    # Authenticity
    authenticity_score: float  # 0-100
    manipulation_detected: bool
    ela_details: Dict

    # Integrity
    file_hashes: Dict[str, str]
    file_size: int

    # Metadata
    comprehensive_metadata: Dict
    exif_data: Dict
    gps_location: Optional[Dict]

    # Device
    device_signature: Optional[str]
    device_verified: bool

    # Verdict
    forensic_verdict: str  # 'authentic', 'suspicious', 'tampered'
    confidence_score: float  # 0-100
    flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BatchAnalysisReport:
    """Batch processing report"""
    total_files: int
    analyzed_files: int
    failed_files: int

    # Results
    authentic_count: int
    suspicious_count: int
    tampered_count: int

    # Sequence integrity
    sequence_complete: bool
    missing_files: List[int]

    # Device consistency
    devices_detected: List[str]
    device_spoofing_detected: bool

    # Individual reports
    file_reports: List[ForensicAnalysisReport]

    # Summary
    analysis_duration: float
    timestamp: str


class ForensicsAnalyzer:
    """
    High-level forensics orchestrator

    Coordinates DBXForensics tools for comprehensive
    automated analysis workflows.
    """

    def __init__(self, toolkit: Optional[DBXForensicsToolkit] = None):
        """
        Initialize analyzer

        Args:
            toolkit: DBXForensicsToolkit instance (creates new if None)
        """
        self.toolkit = toolkit if toolkit else DBXForensicsToolkit()

        # Results directory
        self.results_dir = Path(__file__).parent / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device signatures database
        self.device_signatures: Dict[str, str] = {}
        self._load_device_signatures()

        logger.info("✓ ForensicsAnalyzer initialized")

    def _load_device_signatures(self):
        """Load known device signatures"""
        sig_file = self.results_dir / 'device_signatures.json'
        if sig_file.exists():
            try:
                with open(sig_file, 'r') as f:
                    self.device_signatures = json.load(f)
                logger.info(f"  Loaded {len(self.device_signatures)} device signatures")
            except Exception as e:
                logger.warning(f"Failed to load device signatures: {e}")

    def _save_device_signatures(self):
        """Save device signatures"""
        sig_file = self.results_dir / 'device_signatures.json'
        with open(sig_file, 'w') as f:
            json.dump(self.device_signatures, f, indent=2)

    def register_device_signature(
        self,
        device_id: str,
        sample_images: List[Path]
    ):
        """
        Register device's unique noise signature

        Args:
            device_id: Unique device identifier
            sample_images: List of sample screenshots from device
        """
        logger.info(f"Registering device signature: {device_id}")

        # Extract noise patterns from samples
        noise_patterns = []
        for image_path in sample_images:
            result = self.toolkit.noise_map.analyze(image_path)
            if result.success:
                noise_patterns.append(result.output['noise_signature']['pattern_hash'])

        # Use first pattern as reference (would average in production)
        if noise_patterns:
            self.device_signatures[device_id] = noise_patterns[0]
            self._save_device_signatures()
            logger.info(f"  ✓ Registered signature for {device_id}")

    def analyze_screenshot(
        self,
        image_path: Path,
        expected_device_id: Optional[str] = None
    ) -> ForensicAnalysisReport:
        """
        Comprehensive forensic analysis of single screenshot

        Workflow:
        1. ELA analysis (manipulation detection)
        2. Noise map analysis (device fingerprinting)
        3. Metadata extraction (EXIF, GPS, timestamps)
        4. Hash calculation (integrity verification)
        5. Generate comprehensive report

        Args:
            image_path: Path to screenshot
            expected_device_id: Expected device (for verification)

        Returns:
            ForensicAnalysisReport with complete analysis
        """
        logger.info(f"Analyzing screenshot: {image_path}")
        start_time = datetime.now()

        flags = []
        warnings = []

        # 1. ELA Analysis (manipulation detection)
        logger.info("  [1/4] ELA manipulation detection...")
        ela_result = self.toolkit.ela.analyze(image_path)

        if ela_result.success:
            ela_details = ela_result.output
            authenticity_score = ela_details.get('authenticity_score', 0)
            manipulation_detected = authenticity_score < 70

            if manipulation_detected:
                flags.append('MANIPULATION_DETECTED')
                warnings.append(f"ELA authenticity score low: {authenticity_score}/100")
        else:
            ela_details = {}
            authenticity_score = 0
            manipulation_detected = False
            warnings.append("ELA analysis failed")

        # 2. Noise Map (device fingerprinting)
        logger.info("  [2/4] Device fingerprinting...")
        noise_result = self.toolkit.noise_map.analyze(image_path)

        device_signature = None
        device_verified = False

        if noise_result.success:
            device_signature = noise_result.output['noise_signature']['pattern_hash']

            # Verify against expected device
            if expected_device_id and expected_device_id in self.device_signatures:
                expected_signature = self.device_signatures[expected_device_id]
                device_verified = (device_signature == expected_signature)

                if not device_verified:
                    flags.append('DEVICE_MISMATCH')
                    warnings.append(
                        f"Device signature mismatch! "
                        f"Expected {expected_device_id}, got different signature"
                    )
            else:
                # Check if matches any known device
                for device_id, signature in self.device_signatures.items():
                    if device_signature == signature:
                        warnings.append(f"Matches known device: {device_id}")
                        break

        # 3. Metadata Extraction
        logger.info("  [3/4] Metadata extraction...")
        metadata_result = self.toolkit.metadata.extract(image_path)

        if metadata_result.success:
            comprehensive_metadata = metadata_result.output
            exif_data = comprehensive_metadata.get('exif', {})
            gps_location = None

            if 'gps' in comprehensive_metadata:
                gps_location = comprehensive_metadata['gps']
        else:
            comprehensive_metadata = {}
            exif_data = {}
            gps_location = None
            warnings.append("Metadata extraction failed")

        # 4. Hash Calculation (integrity)
        logger.info("  [4/4] Hash calculation...")
        hash_result = self.toolkit.hash_file.calculate_hashes(
            image_path,
            algorithms=['md5', 'sha1', 'sha256', 'sha512']
        )

        if hash_result.success:
            file_hashes = hash_result.output
        else:
            file_hashes = {}
            warnings.append("Hash calculation failed")

        # Determine verdict
        if len(flags) == 0 and authenticity_score >= 80:
            forensic_verdict = 'authentic'
            confidence_score = authenticity_score
        elif manipulation_detected or 'DEVICE_MISMATCH' in flags:
            forensic_verdict = 'tampered'
            confidence_score = 100 - authenticity_score
        else:
            forensic_verdict = 'suspicious'
            confidence_score = 50.0

        # Build report
        report = ForensicAnalysisReport(
            file_path=str(image_path),
            analysis_timestamp=datetime.now().isoformat(),
            authenticity_score=authenticity_score,
            manipulation_detected=manipulation_detected,
            ela_details=ela_details,
            file_hashes=file_hashes,
            file_size=image_path.stat().st_size,
            comprehensive_metadata=comprehensive_metadata,
            exif_data=exif_data,
            gps_location=gps_location,
            device_signature=device_signature,
            device_verified=device_verified,
            forensic_verdict=forensic_verdict,
            confidence_score=confidence_score,
            flags=flags,
            warnings=warnings
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"  ✓ Analysis complete in {duration:.2f}s")
        logger.info(f"  Verdict: {forensic_verdict.upper()} (confidence: {confidence_score:.1f}%)")

        # Save report
        self._save_report(report, image_path)

        return report

    def batch_analyze(
        self,
        image_paths: List[Path],
        check_sequence: bool = True
    ) -> BatchAnalysisReport:
        """
        Batch analysis of multiple screenshots

        Args:
            image_paths: List of image paths to analyze
            check_sequence: Check for missing files in sequence

        Returns:
            BatchAnalysisReport with aggregate results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Forensic Analysis - {len(image_paths)} files")
        logger.info(f"{'='*60}\n")

        start_time = datetime.now()

        file_reports = []
        failed = 0

        authentic = 0
        suspicious = 0
        tampered = 0

        devices_detected = set()

        # Analyze each file
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{len(image_paths)}] {image_path.name}")

            try:
                report = self.analyze_screenshot(image_path)
                file_reports.append(report)

                # Count verdicts
                if report.forensic_verdict == 'authentic':
                    authentic += 1
                elif report.forensic_verdict == 'suspicious':
                    suspicious += 1
                elif report.forensic_verdict == 'tampered':
                    tampered += 1

                # Track devices
                if report.device_signature:
                    devices_detected.add(report.device_signature)

            except Exception as e:
                logger.error(f"  ✗ Analysis failed: {e}")
                failed += 1

        # Sequence check
        sequence_complete = True
        missing_files = []

        if check_sequence:
            logger.info("\nChecking sequence integrity...")
            missing_files = self._check_sequence_integrity(image_paths)
            sequence_complete = len(missing_files) == 0

            if not sequence_complete:
                logger.warning(f"  ⚠️  Missing {len(missing_files)} files in sequence")

        # Device consistency check
        device_spoofing_detected = len(devices_detected) > 1
        if device_spoofing_detected:
            logger.warning(
                f"  ⚠️  Multiple device signatures detected! "
                f"Possible device spoofing or mixed sources."
            )

        duration = (datetime.now() - start_time).total_seconds()

        # Build report
        report = BatchAnalysisReport(
            total_files=len(image_paths),
            analyzed_files=len(file_reports),
            failed_files=failed,
            authentic_count=authentic,
            suspicious_count=suspicious,
            tampered_count=tampered,
            sequence_complete=sequence_complete,
            missing_files=missing_files,
            devices_detected=list(devices_detected),
            device_spoofing_detected=device_spoofing_detected,
            file_reports=file_reports,
            analysis_duration=duration,
            timestamp=datetime.now().isoformat()
        )

        # Save batch report
        self._save_batch_report(report)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Batch Analysis Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total files:    {report.total_files}")
        logger.info(f"Analyzed:       {report.analyzed_files}")
        logger.info(f"Failed:         {report.failed_files}")
        logger.info(f"")
        logger.info(f"Authentic:      {authentic}")
        logger.info(f"Suspicious:     {suspicious}")
        logger.info(f"Tampered:       {tampered}")
        logger.info(f"")
        logger.info(f"Sequence:       {'✓ Complete' if sequence_complete else '✗ Gaps detected'}")
        logger.info(f"Devices:        {len(devices_detected)}")
        logger.info(f"Duration:       {duration:.2f}s")
        logger.info(f"{'='*60}\n")

        return report

    def compare_screenshots(
        self,
        screenshot_a: Path,
        screenshot_b: Path
    ) -> Dict:
        """
        Compare two screenshots visually

        Args:
            screenshot_a: First screenshot
            screenshot_b: Second screenshot

        Returns:
            Comparison report
        """
        logger.info(f"Comparing screenshots:")
        logger.info(f"  A: {screenshot_a.name}")
        logger.info(f"  B: {screenshot_b.name}")

        result = self.toolkit.ghost.compare(screenshot_a, screenshot_b)

        if result.success:
            diff_score = result.output['difference_score']

            if diff_score < 5:
                verdict = 'identical'
            elif diff_score < 20:
                verdict = 'minor_differences'
            elif diff_score < 50:
                verdict = 'significant_differences'
            else:
                verdict = 'completely_different'

            logger.info(f"  Difference: {diff_score:.1f}% ({verdict})")

            return {
                'screenshot_a': str(screenshot_a),
                'screenshot_b': str(screenshot_b),
                'difference_score': diff_score,
                'verdict': verdict,
                'difference_visualization': result.output.get('difference_visualization')
            }
        else:
            logger.error("  ✗ Comparison failed")
            return {}

    def _check_sequence_integrity(self, image_paths: List[Path]) -> List[int]:
        """Check for missing files in sequence"""
        # Extract sequence numbers from filenames
        # Format: Screenshot_20251117-123456.png -> 20251117123456
        sequence_numbers = []

        for path in image_paths:
            # Simple extraction - would be more robust in production
            import re
            match = re.search(r'(\d{8})[_-](\d{6})', path.name)
            if match:
                seq_num = int(match.group(1) + match.group(2))
                sequence_numbers.append(seq_num)

        if not sequence_numbers:
            return []

        # Check with dbxSeqCheck
        result = self.toolkit.seq_check.check_sequence(sorted(sequence_numbers))

        if result.success:
            return result.output.get('gaps', [])
        else:
            return []

    def _save_report(self, report: ForensicAnalysisReport, image_path: Path):
        """Save individual analysis report"""
        report_dir = self.results_dir / 'individual_reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"{image_path.stem}_report.json"

        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)

    def _save_batch_report(self, report: BatchAnalysisReport):
        """Save batch analysis report"""
        report_dir = self.results_dir / 'batch_reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"batch_analysis_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)

        logger.info(f"Report saved: {report_file}")


# CLI interface
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Forensics Analyzer")
        print("Usage:")
        print("  python3 forensics_analyzer.py analyze <image_path>")
        print("  python3 forensics_analyzer.py batch <directory>")
        print("  python3 forensics_analyzer.py compare <image_a> <image_b>")
        sys.exit(1)

    command = sys.argv[1]
    analyzer = ForensicsAnalyzer()

    if command == 'analyze' and len(sys.argv) >= 3:
        image_path = Path(sys.argv[2])
        report = analyzer.analyze_screenshot(image_path)

        print(f"\nForensic Verdict: {report.forensic_verdict.upper()}")
        print(f"Confidence: {report.confidence_score:.1f}%")
        print(f"Authenticity Score: {report.authenticity_score:.1f}%")

        if report.flags:
            print(f"Flags: {', '.join(report.flags)}")
        if report.warnings:
            print(f"Warnings:")
            for warning in report.warnings:
                print(f"  • {warning}")

    elif command == 'batch' and len(sys.argv) >= 3:
        directory = Path(sys.argv[2])
        image_paths = list(directory.glob('*.png')) + list(directory.glob('*.jpg'))

        report = analyzer.batch_analyze(image_paths)

    elif command == 'compare' and len(sys.argv) >= 4:
        image_a = Path(sys.argv[2])
        image_b = Path(sys.argv[3])

        result = analyzer.compare_screenshots(image_a, image_b)

        if result:
            print(f"\nDifference Score: {result['difference_score']:.1f}%")
            print(f"Verdict: {result['verdict']}")
