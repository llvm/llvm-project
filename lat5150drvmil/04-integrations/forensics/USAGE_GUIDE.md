# DBXForensics Integration - Usage Guide

**LAT5150 DRVMIL - Comprehensive Forensic Analysis System**

Complete guide to using all 9 DBXForensics tools, automated pipelines, and natural language forensic capabilities integrated into LAT5150 intelligence database.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Tool-by-Tool Usage](#tool-by-tool-usage)
4. [High-Level Orchestration](#high-level-orchestration)
5. [Automated Pipelines](#automated-pipelines)
6. [Enhanced Screenshot Intelligence](#enhanced-screenshot-intelligence)
7. [Natural Language Interface](#natural-language-interface)
8. [Python API Examples](#python-api-examples)
9. [Common Workflows](#common-workflows)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# 1. Install Wine (for Windows .exe tools on Linux)
sudo apt-get install wine wine64

# 2. Download DBXForensics tools
# Visit: https://www.dbxforensics.com/Tools/Download
# Place all .exe files in:
/home/user/LAT5150DRVMIL/04-integrations/forensics/tools/

# 3. Verify tools are available
cd /home/user/LAT5150DRVMIL/04-integrations/forensics
python3 dbxforensics_toolkit.py  # Should list all 9 tools
```

### First Forensic Analysis (30 seconds)

```python
#!/usr/bin/env python3
from pathlib import Path
from forensics_analyzer import ForensicsAnalyzer

# Initialize analyzer
analyzer = ForensicsAnalyzer()

# Analyze a screenshot
screenshot = Path("~/screenshots/Screenshot_20251117-101530.png").expanduser()

# Run comprehensive analysis
report = analyzer.analyze_screenshot(screenshot)

print(f"Verdict: {report.forensic_verdict.upper()}")
print(f"Authenticity: {report.authenticity_score:.1f}%")
print(f"Manipulation: {report.manipulation_detected}")
print(f"Device: {report.device_signature or 'Unknown'}")
```

**Output:**
```
✓ ELA analysis complete
✓ Noise map generated
✓ Metadata extracted
✓ Hashes calculated

Verdict: AUTHENTIC
Authenticity: 94.2%
Manipulation: False
Device: device_abc123_signature
```

---

## Architecture Overview

### Component Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                   Natural Language Interface                    │
│  "Is this screenshot authentic?" → forensics_check_authenticity │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               Unified Tactical API (Port 80/5001)               │
│    9 Forensics Capabilities + 24 Existing = 33 Total Caps      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    ForensicsAnalyzer                            │
│      High-level orchestration of all 9 tools                    │
│   - analyze_screenshot() - 4-step comprehensive analysis        │
│   - batch_analyze() - process multiple screenshots             │
│   - compare_screenshots() - visual comparison                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  DBXForensicsToolkit                            │
│           Individual tool wrappers (9 tools)                    │
│  Screenshot │ ELA │ NoiseMap │ Metadata │ HashFile             │
│  SeqCheck │ CsvViewer │ Ghost │ MouseRecorder                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              DBXForensics Windows Tools (.exe)                  │
│                  Wine Compatibility Layer                       │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **Enhanced Screenshot Intelligence**: Automatic forensics during screenshot ingestion
2. **Vector RAG System**: Stores forensic metadata with screenshots
3. **Capability Registry**: 9 forensics capabilities for natural language access
4. **Forensics Knowledge**: AI understanding of forensic concepts
5. **Automated Pipelines**: 4 workflow automations for common tasks

---

## Tool-by-Tool Usage

### 1. dbxScreenshot - Forensic Screenshot Capture

**Purpose**: Capture screenshots with forensic metadata (UTC timestamps, 3 hashes)

**Python API:**
```python
from dbxforensics_toolkit import DBXForensicsToolkit
from pathlib import Path

toolkit = DBXForensicsToolkit()

# Capture full screen
result = toolkit.screenshot.capture(
    output_path=Path("evidence_001.png")
)

if result.success:
    print(f"✓ Screenshot captured: {result.stdout}")
    # Output includes MD5, SHA1, SHA256 hashes
```

**Natural Language:**
```
"Take a forensic screenshot"
"Capture screen with metadata"
"Evidence capture now"
```

**Use Cases:**
- Court-admissible evidence collection
- Automated monitoring with integrity
- Time-stamped incident documentation

---

### 2. dbxELA - Error Level Analysis (Manipulation Detection)

**Purpose**: Detect photoshopped or manipulated areas in JPEG images

**Python API:**
```python
from forensics_analyzer import ForensicsAnalyzer

analyzer = ForensicsAnalyzer()

# Analyze single image
result = analyzer.toolkit.ela.analyze(
    image_path=Path("screenshot.jpg"),
    quality=90  # JPEG quality threshold
)

if result.success:
    parsed = analyzer.toolkit.ela.parse_output(result.stdout, result.stderr)
    print(f"Manipulation detected: {parsed.get('manipulated', False)}")
    print(f"Suspicious regions: {parsed.get('regions', [])}")
```

**Natural Language:**
```
"Is this screenshot authentic?"
"Check for manipulation"
"Detect photoshopping"
"ELA analysis on image.jpg"
```

**Interpretation:**
- **High authenticity (90-100%)**: Likely original, unedited
- **Medium (60-89%)**: Minor editing or compression artifacts
- **Low (<60%)**: Significant manipulation detected

---

### 3. dbxNoiseMap - Device Fingerprinting

**Purpose**: Extract unique noise pattern signature from camera/device sensor

**Python API:**
```python
# Analyze device signature
result = analyzer.toolkit.noise_map.analyze(
    image_path=Path("screenshot.png")
)

if result.success:
    parsed = analyzer.toolkit.noise_map.parse_output(result.stdout, result.stderr)
    signature = parsed.get('signature', '')

    print(f"Device signature: {signature}")

# Register known device signature
analyzer.register_device_signature(
    device_id="phone1",
    sample_images=[
        Path("sample1.png"),
        Path("sample2.png"),
        Path("sample3.png")
    ]
)

# Verify screenshot against known device
report = analyzer.analyze_screenshot(
    image_path=Path("test.png"),
    expected_device_id="phone1"
)

print(f"Device verified: {report.device_verified}")
```

**Natural Language:**
```
"Which device captured this?"
"Device fingerprint analysis"
"Identify camera model"
"Verify this came from Phone 1"
```

**Use Cases:**
- Attribute screenshots to specific devices
- Detect screenshots from unauthorized devices
- Link evidence to known sources

---

### 4. dbxMetadata - Comprehensive Metadata Extraction

**Purpose**: Extract EXIF, GPS, timestamps, camera settings, software info

**Python API:**
```python
result = analyzer.toolkit.metadata.extract(
    file_path=Path("image.jpg"),
    output_format='json'
)

if result.success:
    metadata = analyzer.toolkit.metadata.parse_output(result.stdout, result.stderr)

    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"GPS Location: {metadata.get('gps', 'N/A')}")
    print(f"Camera: {metadata.get('camera_model', 'N/A')}")
    print(f"Software: {metadata.get('software', 'N/A')}")
```

**Natural Language:**
```
"Extract metadata from image.jpg"
"When was this photo taken?"
"Get GPS location from screenshot"
"Show EXIF data"
```

**Common Metadata Fields:**
- **Timestamps**: Creation, modification, EXIF DateTimeOriginal
- **GPS**: Latitude, longitude, altitude
- **Camera**: Make, model, lens, ISO, aperture, shutter speed
- **Software**: Editing applications used

---

### 5. dbxHashFile - Cryptographic Integrity Hashing

**Purpose**: Generate multiple cryptographic hashes for integrity verification

**Python API:**
```python
result = analyzer.toolkit.hash_file.calculate_hashes(
    file_path=Path("evidence.png"),
    algorithms=['sha256', 'sha512', 'sha3-256']
)

if result.success:
    hashes = analyzer.toolkit.hash_file.parse_output(result.stdout, result.stderr)

    print(f"SHA-256: {hashes.get('sha256', '')}")
    print(f"SHA-512: {hashes.get('sha512', '')}")
    print(f"SHA3-256: {hashes.get('sha3-256', '')}")
```

**Natural Language:**
```
"Calculate SHA256 hash"
"Generate integrity hashes"
"Verify file integrity"
"Hash this evidence"
```

**Algorithms Available:**
- CRC32 (fast, weak)
- MD5 (legacy, weak)
- SHA-1 (legacy, weak)
- **SHA-256** (recommended, strong)
- **SHA-512** (recommended, strong)
- **SHA3-256** (recommended, quantum-resistant)

---

### 6. dbxSeqCheck - Sequence Integrity Verification

**Purpose**: Detect missing items, duplicates, or ordering errors in sequences

**Python API:**
```python
# Check screenshot sequence (timestamps)
timestamps = [1731834930, 1731834940, 1731834960]  # Missing 1731834950

result = analyzer.toolkit.seq_check.check_sequence(timestamps)

if result.success:
    parsed = analyzer.toolkit.seq_check.parse_output(result.stdout, result.stderr)

    print(f"Sequence complete: {parsed.get('complete', True)}")
    print(f"Missing items: {parsed.get('missing', [])}")
    print(f"Duplicates: {parsed.get('duplicates', [])}")
```

**Natural Language:**
```
"Check for missing screenshots"
"Verify sequence completeness"
"Detect gaps in timeline"
"Are there duplicates?"
```

**Use Cases:**
- Verify no screenshots deleted from evidence
- Detect gaps in log files
- Validate transaction sequences
- Ensure complete audit trails

---

### 7. dbxCsvViewer - CSV Data Analysis

**Purpose**: Parse CSV files, analyze structured data, export to Excel

**Python API:**
```python
result = analyzer.toolkit.csv_viewer.analyze(
    csv_path=Path("chat_logs.csv"),
    delimiter=','
)

if result.success:
    parsed = analyzer.toolkit.csv_viewer.parse_output(result.stdout, result.stderr)

    print(f"Rows: {parsed.get('row_count', 0)}")
    print(f"Columns: {parsed.get('columns', [])}")
```

**Natural Language:**
```
"Analyze CSV file chat_logs.csv"
"Parse this CSV data"
"Convert CSV to Excel"
"Examine structured data"
```

**Use Cases:**
- Analyze exported chat logs (Telegram, Signal)
- Parse system logs in CSV format
- Examine database exports
- Process structured evidence data

---

### 8. dbxGhost - Visual Screenshot Comparison

**Purpose**: Overlay two screenshots to visually compare changes

**Python API:**
```python
result = analyzer.toolkit.ghost.compare(
    image_a=Path("before.png"),
    image_b=Path("after.png"),
    transparency=50  # 0-100%
)

if result.success:
    print("✓ Comparison visualization generated")
```

**Natural Language:**
```
"Compare these two screenshots"
"What changed between before and after?"
"Show differences visually"
"Overlay screenshots"
```

**Use Cases:**
- Compare UI states before/after events
- Detect subtle visual changes
- Verify screenshot consistency
- Document visual evidence changes

---

### 9. dbxMouseRecorder - Workflow Automation

**Purpose**: Record and replay mouse/keyboard workflows for batch processing

**Python API:**
```python
# Record workflow
result = analyzer.toolkit.mouse_recorder.record_workflow(
    output_script=Path("analysis_workflow.script"),
    duration=60  # seconds
)

# Replay workflow
result = analyzer.toolkit.mouse_recorder.replay_workflow(
    workflow_script=Path("analysis_workflow.script"),
    repeat=10  # repeat 10 times
)
```

**Natural Language:**
```
"Record forensic workflow"
"Automate this analysis"
"Replay workflow 5 times"
```

**Use Cases:**
- Automate repetitive forensic tasks
- Batch process large evidence sets
- Standardize analysis procedures
- Create reproducible workflows

---

## High-Level Orchestration

### ForensicsAnalyzer - Complete Analysis

**4-Step Comprehensive Analysis:**

```python
from forensics_analyzer import ForensicsAnalyzer
from pathlib import Path

analyzer = ForensicsAnalyzer()

# Single comprehensive analysis
report = analyzer.analyze_screenshot(
    image_path=Path("screenshot.png"),
    expected_device_id="phone1"  # Optional
)

# Results include:
print(f"Verdict: {report.forensic_verdict}")  # 'authentic', 'suspicious', 'tampered'
print(f"Authenticity Score: {report.authenticity_score}%")
print(f"Manipulation: {report.manipulation_detected}")
print(f"Device Signature: {report.device_signature}")
print(f"Device Verified: {report.device_verified}")
print(f"Confidence: {report.confidence_score}%")
print(f"SHA-256: {report.file_hashes['sha256']}")
print(f"Flags: {report.flags}")
print(f"Warnings: {report.warnings}")
```

**What Happens:**
1. **ELA Analysis**: Detects manipulation, calculates authenticity score
2. **Noise Map**: Extracts device signature, verifies against expected device
3. **Metadata Extraction**: Gets EXIF, GPS, timestamps, camera info
4. **Hash Calculation**: Generates SHA-256, SHA-512, SHA3-256

**Verdict Logic:**
- **AUTHENTIC**: High authenticity (>85%), no manipulation, device verified
- **SUSPICIOUS**: Medium authenticity (60-85%) OR unverified device
- **TAMPERED**: Low authenticity (<60%) OR manipulation detected

---

### Batch Analysis

```python
# Analyze multiple screenshots
screenshots = [
    Path("screenshot1.png"),
    Path("screenshot2.png"),
    Path("screenshot3.png")
]

batch_report = analyzer.batch_analyze(
    image_paths=screenshots,
    check_sequence=True  # Also verify sequence integrity
)

# Results
print(f"Total: {batch_report.total_analyzed}")
print(f"Authentic: {batch_report.authentic_count}")
print(f"Suspicious: {batch_report.suspicious_count}")
print(f"Tampered: {batch_report.tampered_count}")
print(f"Authenticity Rate: {batch_report.authenticity_rate:.1f}%")

# Sequence check
if batch_report.sequence_check:
    print(f"Sequence Complete: {batch_report.sequence_check.get('complete')}")
    print(f"Missing Items: {batch_report.sequence_check.get('missing', [])}")
```

---

## Automated Pipelines

### Pipeline 1: Evidence Collection

**Automated screenshot capture with forensic metadata:**

```python
from forensics_pipelines import EvidenceCollectionPipeline
from pathlib import Path

# Initialize pipeline
pipeline = EvidenceCollectionPipeline(
    output_dir=Path("~/forensic_evidence").expanduser()
)

# Capture 10 screenshots at 30-second intervals
result = pipeline.execute(
    capture_count=10,
    capture_interval=30,
    region=None  # Full screen (or specify (x, y, width, height))
)

print(f"Execution ID: {result.execution_id}")
print(f"Success: {result.success}")
print(f"Captures: {result.items_success}/{result.items_processed}")

# Chain of custody saved to:
# ~/forensic_evidence/chain_of_custody_<execution_id>.json
```

**Chain of Custody Entry:**
```json
{
  "capture_id": 1,
  "timestamp": "2025-11-17T10:15:30.123456",
  "screenshot_path": "/home/user/forensic_evidence/forensic_capture_20251117_101530_1.png",
  "sha256": "abc123...",
  "sha512": "def456...",
  "sha3_256": "789xyz...",
  "metadata": { ... },
  "captured_by": "LAT5150_ForensicsPipeline",
  "pipeline_execution": "Evidence Collection_20251117_101530"
}
```

---

### Pipeline 2: Authenticity Verification

**Batch verify screenshot authenticity:**

```python
from forensics_pipelines import AuthenticityVerificationPipeline

pipeline = AuthenticityVerificationPipeline()

screenshots = list(Path("~/screenshots").expanduser().glob("*.png"))

result = pipeline.execute(
    screenshot_paths=screenshots,
    quality_threshold=90
)

# View statistics
stats = result.results['statistics']
print(f"Total: {stats['total']}")
print(f"Authentic: {stats['authentic']} ({stats['authenticity_rate']:.1f}%)")
print(f"Tampered: {stats['tampered']} ({stats['tamper_rate']:.1f}%)")

# List tampered files
for screenshot in result.results['tampered']:
    print(f"⚠️  TAMPERED: {screenshot}")
```

---

### Pipeline 3: Incident Investigation

**Complete incident forensic analysis:**

```python
from forensics_pipelines import IncidentInvestigationPipeline

pipeline = IncidentInvestigationPipeline()

# Expected hashes (from original evidence collection)
expected_hashes = {
    "screenshot1.png": "abc123...",
    "screenshot2.png": "def456...",
    "screenshot3.png": "789xyz..."
}

screenshots = [Path(f"screenshot{i}.png") for i in range(1, 4)]

result = pipeline.execute(
    screenshot_paths=screenshots,
    incident_name="Incident_2025_001",
    expected_hashes=expected_hashes
)

# Overall verdict
print(f"Investigation Verdict: {result.results['investigation_verdict']}")
# VERIFIED or COMPROMISED

# Detailed results
print(f"Integrity: {result.results['integrity_check']}")
print(f"Sequence: {result.results['sequence_check']}")
print(f"Authenticity: {result.results['authenticity_check']}")
print(f"Timeline: {len(result.results['timeline'])} entries")
```

---

### Pipeline 4: Continuous Monitoring

**Real-time forensic monitoring:**

```python
from forensics_pipelines import ContinuousMonitoringPipeline

def alert_callback(alert):
    """Called when tampered file detected"""
    print(f"🚨 ALERT: {alert['filename']} - {alert['verdict']}")
    # Send email, Slack notification, etc.

pipeline = ContinuousMonitoringPipeline(
    alert_callback=alert_callback
)

# Monitor directory for 1 hour
result = pipeline.execute(
    watch_directory=Path("~/monitored_screenshots").expanduser(),
    duration_seconds=3600,  # 1 hour
    check_interval=10  # Check every 10 seconds
)

print(f"Files detected: {len(result.results['files_detected'])}")
print(f"Alerts generated: {len(result.results['alerts'])}")
```

---

## Enhanced Screenshot Intelligence

### Automatic Forensic Analysis During Ingestion

```python
from enhanced_screenshot_intelligence import EnhancedScreenshotIntelligence
from pathlib import Path

# Initialize with forensics enabled
intel = EnhancedScreenshotIntelligence(
    enable_forensics=True,
    auto_verify=True,
    alert_on_tampering=True
)

# Ingest screenshot with automatic forensics
result = intel.ingest_screenshot_with_forensics(
    screenshot_path=Path("screenshot.png"),
    device_id="phone1",
    verify_device=True
)

# Results include base ingestion + forensics
print(f"Ingestion: {result.get('status')}")
print(f"Forensics: {result.get('forensics')}")
```

**What Happens Automatically:**
1. Base ingestion (OCR, timeline placement, Vector RAG storage)
2. ELA analysis (manipulation detection)
3. Noise map (device fingerprinting)
4. Metadata extraction
5. Hash calculation
6. Tamper alert generation (if suspicious)
7. Hash chain entry creation

---

### Device Registration with Signature Learning

```python
# Register device and learn its unique noise signature
sample_screenshots = [
    Path("~/phone1_samples/sample1.png").expanduser(),
    Path("~/phone1_samples/sample2.png").expanduser(),
    Path("~/phone1_samples/sample3.png").expanduser()
]

result = intel.register_device_with_signature(
    device_id="phone1",
    device_name="GrapheneOS Phone 1",
    device_type="grapheneos",
    screenshot_path=Path("~/screenshots/phone1").expanduser(),
    sample_screenshots=sample_screenshots
)

print(f"Device registered: {result['device_registered']}")
print(f"Signature learned: {result['forensic_signature']['signature_registered']}")
```

**Future Screenshots Automatically Verified:**
```python
# New screenshot from phone1
intel.ingest_screenshot_with_forensics(
    screenshot_path=Path("new_screenshot.png"),
    device_id="phone1",
    verify_device=True  # Will check against learned signature
)

# If signature doesn't match → Warning in report
```

---

### Incident Integrity Verification

```python
# Create incident from related screenshots
event_ids = ["event_1", "event_2", "event_3"]

incident = intel.create_incident(
    incident_name="Incident_2025_001",
    event_ids=event_ids,
    tags=["investigation", "evidence"]
)

# Verify incident integrity
integrity_report = intel.verify_incident_integrity(incident.incident_id)

print(f"Integrity Verified: {integrity_report['integrity_verified']}")
print(f"Authenticity Stats: {integrity_report['authenticity_stats']}")
print(f"Sequence Check: {integrity_report['sequence_check']}")
print(f"Tampered Events: {integrity_report['tampered_events']}")
```

---

### Batch Ingestion with Forensics

```python
screenshots = list(Path("~/new_evidence").expanduser().glob("*.png"))

batch_result = intel.batch_ingest_with_forensics(
    screenshot_paths=screenshots,
    device_id="phone1",
    check_sequence=True
)

print(f"Total: {batch_result['total']}")
print(f"Success: {batch_result['success']}")
print(f"Authentic: {batch_result['authentic']}")
print(f"Suspicious: {batch_result['suspicious']}")
print(f"Tampered: {batch_result['tampered']}")
print(f"Sequence: {batch_result.get('sequence_check')}")
```

---

## Natural Language Interface

### Via Unified Tactical API (Port 80/5001)

**HTTP API:**
```bash
# Check screenshot authenticity
curl -X POST http://localhost:5001/api/nl \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Check if screenshot.png is authentic",
    "context": {}
  }'
```

**Response:**
```json
{
  "capability_id": "forensics_check_authenticity",
  "result": {
    "authenticity_score": 94.2,
    "manipulation_detected": false,
    "forensic_verdict": "authentic",
    "confidence_score": 92.5,
    "flags": [],
    "warnings": [],
    "message": "Verdict: AUTHENTIC (confidence: 92.5%)"
  }
}
```

### Natural Language Query Examples

**Authenticity Queries:**
- "Is this screenshot real?"
- "Check if image.jpg has been photoshopped"
- "Verify screenshot authenticity"
- "Detect manipulation in evidence.png"
- "ELA analysis on screenshot.jpg"

**Device Attribution:**
- "Which device captured this screenshot?"
- "Identify the camera that took this photo"
- "Device fingerprint for image.png"
- "Verify this came from Phone 1"

**Metadata Extraction:**
- "When was this photo taken?"
- "Extract all metadata from image.jpg"
- "Get GPS location from screenshot"
- "Show EXIF data for evidence.png"
- "What camera took this?"

**Integrity Verification:**
- "Calculate SHA256 hash of evidence.png"
- "Generate integrity hashes"
- "Verify file hasn't been tampered with"
- "Hash chain for this screenshot"

**Sequence Analysis:**
- "Check for missing screenshots"
- "Are there gaps in the timeline?"
- "Verify sequence completeness"
- "Detect duplicate entries"

**Data Analysis:**
- "Analyze CSV file chat_logs.csv"
- "Parse this log data"
- "Convert CSV to Excel"

**Visual Comparison:**
- "Compare before.png and after.png"
- "What changed between these screenshots?"
- "Show visual differences"

**Comprehensive Analysis:**
- "Run full forensic analysis on screenshot.png"
- "Complete forensics for all evidence"
- "Comprehensive analysis on incident screenshots"

---

## Python API Examples

### Example 1: Quick Authenticity Check

```python
#!/usr/bin/env python3
"""Quick screenshot authenticity verification"""

from pathlib import Path
from forensics_analyzer import ForensicsAnalyzer

def check_authenticity(screenshot_path: str) -> dict:
    """Check if screenshot is authentic"""
    analyzer = ForensicsAnalyzer()

    report = analyzer.analyze_screenshot(Path(screenshot_path))

    return {
        'authentic': report.forensic_verdict == 'authentic',
        'score': report.authenticity_score,
        'confidence': report.confidence_score,
        'verdict': report.forensic_verdict,
        'issues': report.flags + report.warnings
    }

# Usage
result = check_authenticity("~/screenshots/screenshot.png")

if result['authentic']:
    print(f"✓ Screenshot is AUTHENTIC ({result['score']:.1f}%)")
else:
    print(f"⚠️  Screenshot is {result['verdict'].upper()}")
    print(f"Issues: {', '.join(result['issues'])}")
```

---

### Example 2: Device Attribution

```python
#!/usr/bin/env python3
"""Identify which device captured screenshots"""

from pathlib import Path
from forensics_analyzer import ForensicsAnalyzer

analyzer = ForensicsAnalyzer()

# Register known devices
devices = {
    'phone1': [Path("phone1_sample1.png"), Path("phone1_sample2.png")],
    'phone2': [Path("phone2_sample1.png"), Path("phone2_sample2.png")]
}

for device_id, samples in devices.items():
    analyzer.register_device_signature(device_id, samples)
    print(f"✓ Registered {device_id}")

# Identify unknown screenshot
unknown = Path("mystery_screenshot.png")
report = analyzer.analyze_screenshot(unknown)

print(f"\nDevice Signature: {report.device_signature}")
print(f"Matches known device: {report.device_verified}")

# Try matching against all known devices
for device_id in devices.keys():
    report = analyzer.analyze_screenshot(unknown, expected_device_id=device_id)
    if report.device_verified:
        print(f"✓ Screenshot came from: {device_id}")
        break
```

---

### Example 3: Batch Evidence Verification

```python
#!/usr/bin/env python3
"""Verify integrity of evidence collection"""

from pathlib import Path
from forensics_analyzer import ForensicsAnalyzer

analyzer = ForensicsAnalyzer()

# Evidence directory
evidence_dir = Path("~/incident_001_evidence").expanduser()
screenshots = sorted(evidence_dir.glob("*.png"))

print(f"Analyzing {len(screenshots)} screenshots...")

# Batch analysis
batch_report = analyzer.batch_analyze(
    image_paths=screenshots,
    check_sequence=True
)

# Report
print(f"\n{'='*60}")
print(f"EVIDENCE VERIFICATION REPORT")
print(f"{'='*60}")
print(f"Total Screenshots: {batch_report.total_analyzed}")
print(f"Authentic: {batch_report.authentic_count} ({batch_report.authenticity_rate:.1f}%)")
print(f"Suspicious: {batch_report.suspicious_count}")
print(f"Tampered: {batch_report.tampered_count}")

if batch_report.sequence_check:
    complete = batch_report.sequence_check.get('complete', False)
    print(f"\nSequence Complete: {'✓ YES' if complete else '✗ NO'}")

    if not complete:
        missing = batch_report.sequence_check.get('missing', [])
        print(f"⚠️  Missing items: {missing}")

# Flag tampered files
for result in batch_report.results:
    if result.forensic_verdict == 'tampered':
        print(f"\n🚨 TAMPERED: {result.screenshot_path.name}")
        print(f"   Authenticity: {result.authenticity_score:.1f}%")
        print(f"   Flags: {', '.join(result.flags)}")
```

---

### Example 4: Forensic Evidence Collection

```python
#!/usr/bin/env python3
"""Collect forensic evidence with chain of custody"""

from pathlib import Path
from forensics_pipelines import EvidenceCollectionPipeline
from datetime import datetime
import json

# Create pipeline
pipeline = EvidenceCollectionPipeline(
    output_dir=Path(f"~/evidence_{datetime.now().strftime('%Y%m%d')}").expanduser()
)

# Collect 20 screenshots over 10 minutes (30s intervals)
print("Starting evidence collection...")

result = pipeline.execute(
    capture_count=20,
    capture_interval=30,
    region=None  # Full screen
)

print(f"\n{'='*60}")
print(f"EVIDENCE COLLECTION COMPLETE")
print(f"{'='*60}")
print(f"Execution ID: {result.execution_id}")
print(f"Success: {result.items_success}/{result.items_processed}")
print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")

# Chain of custody report
custody_file = result.results['chain_of_custody'][0]
print(f"\nChain of custody saved to:")
print(f"{pipeline.output_dir}/chain_of_custody_{result.execution_id}.json")

# Verify first entry
first_entry = result.results['chain_of_custody'][0]
print(f"\nFirst Capture:")
print(f"  Timestamp: {first_entry['timestamp']}")
print(f"  SHA-256: {first_entry['sha256'][:16]}...")
print(f"  SHA-512: {first_entry['sha512'][:16]}...")
```

---

## Common Workflows

### Workflow A: Incident Response Evidence Collection

**Scenario**: Collect forensically-sound evidence during incident

```python
from enhanced_screenshot_intelligence import EnhancedScreenshotIntelligence
from forensics_pipelines import EvidenceCollectionPipeline
from pathlib import Path

# 1. Automated evidence capture
pipeline = EvidenceCollectionPipeline()
collection_result = pipeline.execute(capture_count=50, capture_interval=60)

# 2. Ingest into intelligence database with forensics
intel = EnhancedScreenshotIntelligence(
    enable_forensics=True,
    alert_on_tampering=True
)

screenshots = [Path(c['screenshot']) for c in collection_result.results['captures']]

batch_result = intel.batch_ingest_with_forensics(
    screenshot_paths=screenshots,
    device_id="incident_collection_system",
    check_sequence=True
)

# 3. Create incident
event_ids = [r['doc_id'] for r in batch_result['files'] if r['result'].get('status') == 'success']

incident = intel.create_incident(
    incident_name="Incident_2025_001",
    event_ids=event_ids,
    tags=["incident_response", "forensic_evidence"]
)

# 4. Verify integrity
integrity_report = intel.verify_incident_integrity(incident.incident_id)

print(f"Evidence Collected: {len(screenshots)}")
print(f"Incident Created: {incident.incident_id}")
print(f"Integrity Verified: {integrity_report['integrity_verified']}")
```

---

### Workflow B: Historical Evidence Verification

**Scenario**: Verify integrity of existing evidence collection

```python
from forensics_pipelines import IncidentInvestigationPipeline
from pathlib import Path
import json

# Load original chain of custody (from collection)
with open("chain_of_custody_original.json") as f:
    original_custody = json.load(f)

# Build expected hashes
expected_hashes = {
    entry['screenshot_path'].split('/')[-1]: entry['sha256']
    for entry in original_custody
}

# Current evidence files
evidence_dir = Path("~/evidence_archive").expanduser()
screenshots = list(evidence_dir.glob("*.png"))

# Investigate
pipeline = IncidentInvestigationPipeline()

result = pipeline.execute(
    screenshot_paths=screenshots,
    incident_name="Historical_Verification",
    expected_hashes=expected_hashes
)

# Report
verdict = result.results['investigation_verdict']

if verdict == 'VERIFIED':
    print("✓ Evidence integrity VERIFIED")
    print("  All hashes match")
    print("  No tampering detected")
    print("  Sequence complete")
else:
    print("⚠️  Evidence COMPROMISED")

    integrity = result.results['integrity_check']
    if integrity['failed'] > 0:
        print(f"  {integrity['failed']} files have mismatched hashes")

    auth = result.results['authenticity_check']
    if auth['tampered'] > 0:
        print(f"  {auth['tampered']} files show signs of manipulation")
        print(f"  Tampered: {', '.join(auth['tampered_files'])}")
```

---

### Workflow C: Continuous Monitoring with Alerts

**Scenario**: Monitor directory for tampered screenshots in real-time

```python
from forensics_pipelines import ContinuousMonitoringPipeline
from pathlib import Path
import smtplib
from email.message import EmailMessage

def send_alert_email(alert):
    """Send email alert for tampered screenshot"""
    msg = EmailMessage()
    msg.set_content(f"""
    FORENSIC ALERT: Tampered Screenshot Detected

    File: {alert['filename']}
    Time: {alert['timestamp']}
    Verdict: {alert['verdict'].upper()}
    Flags: {', '.join(alert['flags'])}

    Immediate investigation required.
    """)

    msg['Subject'] = f"🚨 FORENSIC ALERT: {alert['filename']}"
    msg['From'] = "forensics@example.com"
    msg['To'] = "security-team@example.com"

    # Send email
    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)

    print(f"📧 Alert email sent for {alert['filename']}")

# Start monitoring
pipeline = ContinuousMonitoringPipeline(
    alert_callback=send_alert_email
)

print("Starting continuous monitoring...")
print("Watching: ~/monitored_screenshots")
print("Press Ctrl+C to stop")

try:
    result = pipeline.execute(
        watch_directory=Path("~/monitored_screenshots").expanduser(),
        duration_seconds=86400,  # 24 hours
        check_interval=5  # Check every 5 seconds
    )
except KeyboardInterrupt:
    print("\nMonitoring stopped")

print(f"\nFiles analyzed: {result.items_processed}")
print(f"Alerts generated: {len(result.results['alerts'])}")
```

---

## Troubleshooting

### Wine Compatibility Issues

**Problem**: Tools fail to execute with Wine errors

**Solution**:
```bash
# Install Wine dependencies
sudo apt-get install wine wine64 wine32

# Configure Wine
winecfg

# Test tool execution
cd /home/user/LAT5150DRVMIL/04-integrations/forensics/tools
wine dbxScreenshot.exe --help

# If still failing, check Wine version
wine --version  # Should be 6.0 or higher
```

---

### Tool Not Found Errors

**Problem**: `Tool executable not found: dbxELA.exe`

**Solution**:
```python
from dbxforensics_toolkit import DBXForensicsToolkit
from pathlib import Path

# Manually specify tools directory
toolkit = DBXForensicsToolkit(
    tools_dir=Path("/home/user/LAT5150DRVMIL/04-integrations/forensics/tools")
)

# Verify tool paths
print(f"Screenshot tool: {toolkit.screenshot.tool_path}")
print(f"ELA tool: {toolkit.ela.tool_path}")

# Check file exists
assert toolkit.ela.tool_path.exists(), "ELA tool not found!"
```

---

### Performance Optimization

**Problem**: Batch analysis is slow for large evidence sets

**Solution**: Use parallel processing (future enhancement)

```python
# Current: Sequential processing
# analyzer.batch_analyze(screenshots)  # Slow for 100+ images

# Recommended: Process in smaller batches
from pathlib import Path

screenshots = list(Path("evidence").glob("*.png"))

batch_size = 10
for i in range(0, len(screenshots), batch_size):
    batch = screenshots[i:i+batch_size]
    result = analyzer.batch_analyze(batch, check_sequence=False)
    print(f"Processed batch {i//batch_size + 1}: {result.authentic_count} authentic")
```

**Future Enhancement**: GPU-accelerated ELA and parallel processing

---

### Memory Issues with Large Evidence Sets

**Problem**: Out of memory when processing 1000+ screenshots

**Solution**:
```python
# Don't load all results in memory
from forensics_analyzer import ForensicsAnalyzer
import json

analyzer = ForensicsAnalyzer()
screenshots = list(Path("evidence").glob("*.png"))

# Stream results to file
with open("forensic_results.jsonl", 'w') as f:
    for screenshot in screenshots:
        report = analyzer.analyze_screenshot(screenshot)

        # Write to file immediately
        f.write(json.dumps({
            'file': screenshot.name,
            'verdict': report.forensic_verdict,
            'score': report.authenticity_score
        }) + '\n')

        # Free memory
        del report

print("✓ Results saved to forensic_results.jsonl")
```

---

## Advanced Topics

### Custom Pipeline Creation

```python
from forensics_pipelines import ForensicsPipeline
from pathlib import Path

class CustomForensicsPipeline(ForensicsPipeline):
    """Custom pipeline for specific workflow"""

    def __init__(self):
        super().__init__("Custom Pipeline")

    def execute(self, *args, **kwargs):
        self._start_execution()

        # Your custom workflow here
        results = {}
        errors = []

        # ...

        self._end_execution(success=True)

        return PipelineResult(
            pipeline_name=self.name,
            execution_id=self.execution_id,
            start_time=self.start_time,
            end_time=self.end_time,
            success=True,
            items_processed=0,
            items_success=0,
            items_failed=0,
            results=results,
            errors=errors,
            warnings=[]
        )
```

---

## Summary

LAT5150 now has comprehensive forensic analysis capabilities:

✅ **9 DBXForensics Tools** fully integrated with Wine compatibility
✅ **High-Level Orchestration** via ForensicsAnalyzer
✅ **4 Automated Pipelines** for common workflows
✅ **Enhanced Screenshot Intelligence** with automatic forensics
✅ **Natural Language Access** via 9 capabilities and 50+ triggers
✅ **AI Knowledge Integration** for intelligent recommendations
✅ **Chain of Custody** support with cryptographic integrity
✅ **Real-Time Monitoring** with tamper alerts
✅ **Batch Processing** for large evidence sets

**Next Steps**:
1. Download DBXForensics tools to `tools/` directory
2. Test with sample screenshot: `python3 forensics_analyzer.py`
3. Register your devices with signature learning
4. Start using natural language queries via Unified API
5. Explore automated pipelines for your workflows

**Questions or Issues?**
- Check FORENSICS_INTEGRATION_PLAN.md for architecture details
- Check CURRENT_SYSTEM_ANALYSIS.md for gap analysis
- Review code comments in all modules
- Test with provided examples

**LAT5150 DRVMIL Forensics System - Ready for Production** 🔬
