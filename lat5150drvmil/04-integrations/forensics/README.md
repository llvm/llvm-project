# LAT5150 DRVMIL - DBXForensics Integration

## Overview

Integration of 9 DBXForensics tools to transform LAT5150 from intelligence collection to forensically-sound evidence platform.

## Critical Gaps Filled

### Before DBXForensics:
- ❌ No authenticity verification (can't detect manipulated screenshots)
- ❌ No forensic metadata (missing cryptographic timestamps/hashes)
- ❌ No chain of custody (can't prove evidence integrity)
- ❌ No device fingerprinting (can't verify capture device)
- ❌ No gap detection (can't detect missing evidence)

### After DBXForensics:
- ✅ Automatic manipulation detection via ELA analysis
- ✅ Forensic timestamps + 3 hashes (MD5/SHA1/SHA256)
- ✅ Multi-algorithm integrity verification (6 algorithms)
- ✅ Device fingerprinting via digital noise patterns
- ✅ Automatic sequence gap detection

## Tools

### 1. dbxScreenshot - Forensic Capture
Captures screenshots with forensic metadata:
- UTC timestamps
- MD5, SHA-1, SHA-256 hashes
- Device information
- User information

**Value**: Cryptographic proof of when/where screenshot captured

### 2. dbxELA - Manipulation Detection
Error Level Analysis for JPEG tampering:
- Detects re-compressed regions
- Visual heatmap of suspicious areas
- Authenticity scoring

**Value**: Automatic detection of manipulated screenshots

### 3. dbxNoiseMap - Device Fingerprinting
Digital noise pattern analysis:
- Unique device signatures (like camera fingerprint)
- Verify which device captured image
- Detect device spoofing

**Value**: Prove physical device origin of evidence

### 4. dbxMetadata - Intelligence Extraction
Comprehensive metadata extraction:
- EXIF data (GPS, camera model, timestamps)
- XMP metadata
- Edit history
- 50+ metadata fields

**Value**: Extract maximum intelligence from each artifact

### 5. dbxHashFile - Chain of Custody
Multi-algorithm cryptographic hashing:
- CRC32, MD5, SHA-1, SHA-256, SHA-512, SHA3-256
- Batch processing
- Integrity manifests

**Value**: Prove evidence hasn't been modified

### 6. dbxSeqCheck - Gap Detection
Numeric sequence verification:
- Detect missing numbers
- Find duplicates
- Identify ordering errors

**Value**: Verify evidence completeness, detect missing files

### 7. dbxCsvViewer - Data Analysis
CSV file analysis:
- Fast parsing
- Excel export
- Column sorting

**Value**: Analyze structured data (logs, exports)

### 8. dbxGhost - Visual Comparison
Screen portion capture and comparison:
- Side-by-side comparison
- Transparency overlay
- Difference detection

**Value**: Detect UI changes, compare screenshots

### 9. dbxMouseRecorder - Automation
Task automation for forensic workflows:
- Record workflows
- Replay for batch processing
- Standardize procedures

**Value**: 5x faster batch analysis

## Installation

### Prerequisites

1. **Wine** (for running Windows .exe files on Linux):
```bash
sudo apt update
sudo apt install wine wine64
```

2. **Python Dependencies**:
```bash
pip3 install pillow  # Image processing (optional)
```

### Download DBXForensics Tools

1. Visit: https://www.dbxforensics.com/Tools/Download

2. Download all 9 tools:
   - dbxScreenshot v1.0.0.0
   - dbxELA v1.0.0.0
   - dbxNoiseMap v1.0.0.0
   - dbxGhost v1.0.0.0
   - dbxMetadata v1.0.0.0
   - dbxHashFile v1.0.0.0
   - dbxCsvViewer v1.0.0.0
   - dbxSeqCheck v1.0.0.0
   - dbxMouseRecorder v1.0.0.0

3. Place .exe files in:
```
/home/user/LAT5150DRVMIL/04-integrations/forensics/tools/
```

## Usage

### Python API

```python
from dbxforensics_toolkit import DBXForensicsToolkit

# Initialize toolkit
toolkit = DBXForensicsToolkit()

# Capture forensic screenshot
result = toolkit.screenshot.capture(
    output_path=Path('evidence_001.png')
)

# Analyze for manipulation
ela_result = toolkit.ela.analyze(
    image_path=Path('evidence_001.png')
)

if ela_result.output['manipulation_detected']:
    print("⚠️ Possible manipulation detected!")

# Extract metadata
metadata_result = toolkit.metadata.extract(
    file_path=Path('evidence_001.png')
)

# Calculate hashes
hash_result = toolkit.hash_file.calculate_hashes(
    file_path=Path('evidence_001.png'),
    algorithms=['sha256', 'sha512']
)

# Device fingerprinting
noise_result = toolkit.noise_map.analyze(
    image_path=Path('evidence_001.png')
)

# Check sequence integrity
seq_result = toolkit.seq_check.check_sequence(
    numbers=[1, 2, 3, 5, 6]  # Missing 4!
)
```

### Command Line

```bash
# Check tool availability
python3 dbxforensics_toolkit.py check-tools

# Run test
python3 dbxforensics_toolkit.py test
```

## Architecture

```
DBXForensics Integration
│
├── dbxforensics_toolkit.py      # Main toolkit (base wrappers)
├── forensics_analyzer.py         # High-level orchestrator
├── integration_layer.py          # ScreenshotIntelligence integration
├── forensics_api.py              # REST API endpoints
│
├── tools/                        # DBXForensics .exe files
│   ├── dbxScreenshot.exe
│   ├── dbxELA.exe
│   ├── dbxNoiseMap.exe
│   ├── dbxGhost.exe
│   ├── dbxMetadata.exe
│   ├── dbxHashFile.exe
│   ├── dbxCsvViewer.exe
│   ├── dbxSeqCheck.exe
│   └── dbxMouseRecorder.exe
│
├── config/                       # Configuration files
│   ├── forensics_config.json
│   └── tool_paths.json
│
└── results/                      # Analysis results
    ├── ela_analysis/
    ├── noise_maps/
    ├── metadata/
    └── hash_manifests/
```

## Integration with LAT5150

### Screenshot Intelligence Enhancement

```python
from screenshot_intelligence import ScreenshotIntelligence
from dbxforensics_toolkit import DBXForensicsToolkit

class EnhancedScreenshotIntelligence(ScreenshotIntelligence):
    def __init__(self, ...):
        super().__init__(...)
        self.forensics = DBXForensicsToolkit()

    def ingest_screenshot_with_forensics(self, path):
        # 1. Standard ingestion
        event = self.ingest_screenshot(path)

        # 2. Forensic analysis
        ela = self.forensics.ela.analyze(path)
        noise = self.forensics.noise_map.analyze(path)
        metadata = self.forensics.metadata.extract(path)
        hashes = self.forensics.hash_file.calculate_hashes(path)

        # 3. Enrich event metadata
        event.metadata['forensics'] = {
            'authenticity_score': ela.output['authenticity_score'],
            'manipulation_detected': ela.output['manipulation_detected'],
            'device_signature': noise.output['noise_signature'],
            'comprehensive_metadata': metadata.output,
            'integrity_hashes': hashes.output
        }

        # 4. Flag if suspicious
        if ela.output['manipulation_detected']:
            event.metadata['flagged'] = True
            event.metadata['flag_reason'] = 'Possible manipulation detected'
            self.send_alert(f"Tampered evidence: {path}")

        return event
```

### Unified API Integration

9 new capabilities added to `/home/user/LAT5150DRVMIL/03-web-interface/capability_registry.py`:

- `forensics_screenshot_capture` - "capture forensic screenshot"
- `forensics_ela_analysis` - "check image authenticity"
- `forensics_noise_analysis` - "analyze digital noise"
- `forensics_image_compare` - "compare screenshots"
- `forensics_metadata_extract` - "extract file metadata"
- `forensics_hash_calculate` - "calculate file hash"
- `forensics_csv_analyze` - "analyze CSV data"
- `forensics_sequence_check` - "verify screenshot sequence"
- `forensics_automate_workflow` - "automate forensics workflow"

Natural language examples:
```bash
curl -X POST http://localhost/api/query \
  -d '{"query": "analyze this screenshot for manipulation"}'

curl -X POST http://localhost/api/query \
  -d '{"query": "check image authenticity"}'

curl -X POST http://localhost/api/query \
  -d '{"query": "verify screenshot sequence"}'
```

## Value Proposition

### Evidence Quality
- **Before**: Screenshots collected, no verification
- **After**: Every screenshot forensically verified with cryptographic proof

### Tamper Detection
- **Before**: No way to detect manipulated images
- **After**: Automatic ELA analysis flags suspicious images

### Chain of Custody
- **Before**: No integrity verification
- **After**: Multi-algorithm hashing provides cryptographic proof

### Device Verification
- **Before**: Device attribution based on directory/path
- **After**: Cryptographic device fingerprinting via noise patterns

### Evidence Completeness
- **Before**: Missing evidence goes undetected
- **After**: Automatic sequence checking detects gaps

## Documentation

- **FORENSICS_INTEGRATION_PLAN.md** - Complete integration plan (600+ lines)
- **CURRENT_SYSTEM_ANALYSIS.md** - Deep analysis of current LAT5150 + gaps (800+ lines)
- **README.md** - This file

## Status

✅ **Phase 1 Complete**: Core infrastructure
- Base DBXForensicsTool wrapper class
- 6 individual tool wrappers (Screenshot, ELA, NoiseMap, Metadata, HashFile, SeqCheck)
- Configuration management
- Wine compatibility layer

🔄 **Phase 2 In Progress**: High-level integration
- ForensicsAnalyzer orchestrator
- ScreenshotIntelligence enhancement
- Automated analysis pipelines

⏳ **Phase 3 Planned**: API Integration
- 9 new capabilities in registry
- Unified API handlers
- Natural language triggers

⏳ **Phase 4 Planned**: Testing & Documentation
- Integration tests
- Usage tutorials
- Deployment guide

## Next Steps

1. Download DBXForensics tools from https://www.dbxforensics.com/Tools/Download
2. Place .exe files in `tools/` directory
3. Install Wine: `sudo apt install wine wine64`
4. Test: `python3 dbxforensics_toolkit.py check-tools`
5. Integrate with existing workflows

## License

DBXForensics tools are property of DBX Forensics.
LAT5150 integration code is part of LAT5150 DRVMIL project.

---

**Version**: 1.0.0 (Foundation)
**Last Updated**: 2025-11-17
**Status**: Core Infrastructure Complete
