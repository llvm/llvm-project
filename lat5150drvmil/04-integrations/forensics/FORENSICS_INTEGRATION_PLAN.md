# LAT5150 DRVMIL - DBXForensics Integration Plan

## Overview

Integration of 9 DBXForensics tools into LAT5150 intelligence database system for comprehensive forensic analysis of screenshots, images, files, and data artifacts.

## Current System Analysis

### Existing Screenshot Intelligence
- **Location**: `/home/user/LAT5150DRVMIL/04-integrations/rag_system/screenshot_intelligence.py`
- **Features**:
  - Screenshot ingestion with OCR
  - Device registration and tracking
  - Chat log correlation (Telegram, Signal)
  - Timeline reconstruction
  - Event clustering and linking
  - Incident grouping
  - Vector RAG integration for storage

### Existing Components
- **Vector RAG System**: Document storage and retrieval
- **Enhanced AI Engine**: Vision and NL capabilities
- **Unified Tactical API**: Natural language interface (port 80)
- **Capability Registry**: 24 registered capabilities
- **DSMIL Hardware Access**: Device enumeration
- **Agent Orchestrator**: Multi-agent coordination

---

## DBXForensics Tools

### 1. Acquisition

#### **dbxScreenshot**
- **Purpose**: Forensic screenshot capture with comprehensive metadata
- **Metadata**: UTC timestamps, MD5/SHA1/SHA-256 hashes, user info, device identifiers
- **Integration Point**: Enhance existing screenshot ingestion pipeline
- **Use Cases**:
  - Forensically-sound evidence capture
  - Tamper-proof screenshot collection
  - Audit trail generation

### 2. Analysis Tools

#### **dbxCsvViewer**
- **Purpose**: CSV file analysis and Excel export
- **Features**: Custom delimiters, column sorting, fast data loading
- **Integration Point**: Data artifact analysis
- **Use Cases**:
  - Analyze exported chat logs
  - Parse system logs
  - Examine structured data artifacts

#### **dbxELA (Error Level Analysis)**
- **Purpose**: JPEG manipulation detection
- **Method**: Error level analysis visualization
- **Integration Point**: Screenshot authenticity verification
- **Use Cases**:
  - Detect manipulated screenshots
  - Verify image authenticity
  - Identify composite images
  - Evidence integrity verification

#### **dbxGhost**
- **Purpose**: Screen portion capture and comparison
- **Features**: Movable/resizable windows, transparency, overlay
- **Integration Point**: Visual comparison pipeline
- **Use Cases**:
  - Compare screenshots across time
  - Detect UI changes
  - Verify consistency

#### **dbxMetadata**
- **Purpose**: Comprehensive file and metadata extraction
- **Supports**: File system info, internal metadata, various formats
- **Integration Point**: Evidence metadata enrichment
- **Use Cases**:
  - Extract EXIF data from screenshots
  - Analyze file timestamps
  - Gather device information

#### **dbxNoiseMap**
- **Purpose**: Digital noise pattern analysis
- **Method**: Spatial noise distribution visualization
- **Integration Point**: Advanced image forensics
- **Use Cases**:
  - Detect cloned regions
  - Identify camera/device fingerprints
  - Reveal hidden modifications

#### **dbxSeqCheck**
- **Purpose**: Numeric sequence integrity verification
- **Features**: Detect missing numbers, duplicates, ordering errors
- **Integration Point**: Data integrity validation
- **Use Cases**:
  - Verify screenshot sequence completeness
  - Detect missing evidence
  - Validate log continuity

### 3. Reporting

#### **dbxHashFile**
- **Purpose**: Multi-algorithm cryptographic hashing
- **Algorithms**: CRC32, MD5, SHA-1, SHA-256, SHA-512, SHA3-256
- **Integration Point**: Evidence integrity chain
- **Use Cases**:
  - Generate hash manifests
  - Verify file integrity
  - Create chain of custody

### 4. Utility

#### **dbxMouseRecorder**
- **Purpose**: Task automation for forensic workflows
- **Features**: Mouse pointer control, reliable automation
- **Integration Point**: Workflow automation
- **Use Cases**:
  - Automate repetitive analysis tasks
  - Batch processing screenshots
  - Standardize workflows

---

## Integration Architecture

### Directory Structure
```
/home/user/LAT5150DRVMIL/
├── 04-integrations/
│   └── forensics/
│       ├── FORENSICS_INTEGRATION_PLAN.md  (this document)
│       ├── dbxforensics_toolkit.py        (main integration)
│       ├── forensics_analyzer.py          (analysis orchestrator)
│       ├── image_forensics.py             (ELA, NoiseMap, Ghost)
│       ├── metadata_extractor.py          (Metadata, HashFile)
│       ├── data_analyzer.py               (CsvViewer, SeqCheck)
│       ├── screenshot_capture.py          (dbxScreenshot wrapper)
│       ├── automation_engine.py           (MouseRecorder)
│       ├── forensics_api.py               (REST endpoints)
│       ├── tools/                         (DBXForensics binaries)
│       │   ├── dbxScreenshot.exe
│       │   ├── dbxELA.exe
│       │   ├── dbxNoiseMap.exe
│       │   ├── dbxGhost.exe
│       │   ├── dbxMetadata.exe
│       │   ├── dbxHashFile.exe
│       │   ├── dbxCsvViewer.exe
│       │   ├── dbxSeqCheck.exe
│       │   └── dbxMouseRecorder.exe
│       └── config/
│           ├── forensics_config.json
│           └── tool_paths.json
```

### Component Design

#### 1. **ForensicsToolkit** (dbxforensics_toolkit.py)
Core wrapper class managing all 9 DBXForensics tools:
- Tool path management
- Process execution
- Result parsing
- Error handling
- Wine/Windows compatibility layer

#### 2. **ForensicsAnalyzer** (forensics_analyzer.py)
High-level orchestrator integrating with ScreenshotIntelligence:
- Automatic analysis pipeline
- Multi-tool workflows
- Results aggregation
- Intelligence database integration

#### 3. **ImageForensics** (image_forensics.py)
Specialized image analysis:
- ELA analysis for manipulation detection
- Noise map generation and analysis
- Image comparison (Ghost)
- Visual forensics reporting

#### 4. **MetadataExtractor** (metadata_extractor.py)
File and metadata extraction:
- Comprehensive metadata parsing
- Hash calculation (all 6 algorithms)
- EXIF/XMP extraction
- Timestamp normalization

#### 5. **DataAnalyzer** (data_analyzer.py)
Structured data analysis:
- CSV parsing and analysis
- Sequence verification
- Data integrity checks
- Export to intelligence database

#### 6. **ScreenshotCapture** (screenshot_capture.py)
Forensic screenshot acquisition:
- Automated capture with dbxScreenshot
- Metadata preservation
- Automatic ingestion to intelligence database
- Hash chain generation

#### 7. **AutomationEngine** (automation_engine.py)
Workflow automation:
- dbxMouseRecorder integration
- Batch processing pipelines
- Scheduled analysis tasks
- Workflow scripting

#### 8. **ForensicsAPI** (forensics_api.py)
REST API endpoints for unified integration:
- Natural language query processing
- Tool execution endpoints
- Results retrieval
- Status monitoring

---

## Integration with Existing Systems

### 1. Enhanced Screenshot Intelligence
```python
class EnhancedScreenshotIntelligence(ScreenshotIntelligence):
    def __init__(self, ...):
        super().__init__(...)
        self.forensics = ForensicsAnalyzer()

    def ingest_screenshot_with_forensics(self, screenshot_path):
        # 1. Original ingestion
        event = self.ingest_screenshot(screenshot_path)

        # 2. Forensic analysis
        analysis = self.forensics.analyze_screenshot(screenshot_path)

        # 3. Enrichment
        event.metadata['forensics'] = {
            'authenticity_score': analysis.ela_score,
            'manipulation_detected': analysis.manipulation_detected,
            'noise_pattern': analysis.noise_signature,
            'file_hash': analysis.hash_sha256,
            'metadata': analysis.exif_data
        }

        return event
```

### 2. Unified Tactical API Integration
Add 9 new capabilities to capability_registry.py:
- `forensics_screenshot_capture` - Forensic screenshot with metadata
- `forensics_ela_analysis` - JPEG manipulation detection
- `forensics_noise_analysis` - Digital noise pattern analysis
- `forensics_image_compare` - Visual comparison (Ghost)
- `forensics_metadata_extract` - Comprehensive metadata extraction
- `forensics_hash_calculate` - Multi-algorithm hashing
- `forensics_csv_analyze` - CSV data analysis
- `forensics_sequence_check` - Numeric sequence verification
- `forensics_automate_workflow` - Workflow automation

### 3. Natural Language Triggers
```python
# Examples:
"analyze this screenshot for manipulation"
"check image authenticity"
"extract metadata from files"
"calculate SHA256 hash"
"compare these two screenshots"
"detect digital noise patterns"
"verify screenshot sequence"
"analyze CSV data"
```

### 4. Integration with Red Team Benchmark
Add forensics analysis to benchmark results:
- Hash verification of benchmark outputs
- Metadata extraction from test artifacts
- Sequence verification of test runs
- Data integrity validation

---

## Workflows

### Workflow 1: Forensic Screenshot Analysis Pipeline
```
1. Capture screenshot with dbxScreenshot
   → Metadata: UTC timestamp, hashes, device info

2. Ingest to ScreenshotIntelligence
   → OCR, device association, timeline placement

3. Authenticity verification (dbxELA)
   → Manipulation detection, authenticity score

4. Noise analysis (dbxNoiseMap)
   → Device fingerprinting, modification detection

5. Metadata extraction (dbxMetadata)
   → EXIF data, timestamps, camera info

6. Hash calculation (dbxHashFile)
   → SHA-256, SHA-512, SHA3-256 for chain of custody

7. Store enriched event in Vector RAG
   → Full forensic metadata preserved

8. Generate forensic report
   → PDF with all analysis results
```

### Workflow 2: Incident Investigation
```
1. Load incident from ScreenshotIntelligence
   → Get all related screenshots and events

2. Verify integrity (dbxHashFile)
   → Confirm no tampering since acquisition

3. Sequence check (dbxSeqCheck)
   → Verify no missing screenshots

4. Batch ELA analysis
   → Identify any manipulated images

5. Timeline reconstruction
   → Metadata-based temporal ordering

6. Generate investigation report
   → Comprehensive forensic summary
```

### Workflow 3: Evidence Collection
```
1. Automated capture (dbxScreenshot + dbxMouseRecorder)
   → Scheduled captures with automation

2. Real-time ingestion
   → Immediate forensic analysis

3. Continuous monitoring
   → Detect anomalies (manipulation, gaps)

4. Alert generation
   → Notify on integrity issues

5. Chain of custody
   → Cryptographic hash trail
```

---

## Technical Implementation

### Python Wrapper Template
```python
class DBXForensicsTool:
    def __init__(self, tool_exe_path):
        self.tool_path = Path(tool_exe_path)
        self.wine_available = shutil.which('wine') is not None

    def execute(self, *args, **kwargs):
        """Execute tool with wine if on Linux"""
        if os.name != 'nt' and self.wine_available:
            cmd = ['wine', str(self.tool_path)] + list(args)
        else:
            cmd = [str(self.tool_path)] + list(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        return self.parse_result(result)

    def parse_result(self, result):
        """Parse tool output"""
        # Tool-specific parsing
        pass
```

### Configuration Management
```json
{
  "tools": {
    "dbxScreenshot": {
      "path": "/home/user/LAT5150DRVMIL/04-integrations/forensics/tools/dbxScreenshot.exe",
      "enabled": true,
      "timeout": 60
    },
    "dbxELA": {
      "path": "/home/user/LAT5150DRVMIL/04-integrations/forensics/tools/dbxELA.exe",
      "enabled": true,
      "quality_threshold": 90
    },
    ...
  },
  "analysis": {
    "auto_ela": true,
    "auto_noise_map": true,
    "auto_metadata": true,
    "auto_hash": true,
    "hash_algorithms": ["sha256", "sha512", "sha3-256"]
  },
  "storage": {
    "results_dir": "/home/user/LAT5150DRVMIL/04-integrations/forensics/results",
    "cache_enabled": true
  }
}
```

---

## Benefits

### For Intelligence Database
1. **Enhanced Evidence Quality**: Forensically-sound capture and analysis
2. **Authenticity Verification**: Detect manipulated screenshots
3. **Integrity Chain**: Cryptographic hash trail
4. **Rich Metadata**: Comprehensive file and device information
5. **Automated Workflows**: Batch processing and analysis

### For Red Team Operations
1. **Evidence Integrity**: Verify benchmark test results
2. **Data Validation**: Check sequence and completeness
3. **Artifact Analysis**: Parse and analyze test outputs
4. **Report Generation**: Forensic-quality documentation

### For System Operations
1. **Audit Trail**: Comprehensive logging and hashing
2. **Tamper Detection**: Identify modified files
3. **Device Fingerprinting**: Noise pattern analysis
4. **Workflow Automation**: Standardized processes

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
- [ ] Create forensics directory structure
- [ ] Download and configure DBXForensics tools
- [ ] Implement DBXForensicsTool base wrapper
- [ ] Configuration management
- [ ] Wine compatibility layer

### Phase 2: Individual Tool Wrappers (Days 3-4)
- [ ] dbxScreenshot wrapper and integration
- [ ] dbxELA wrapper and analysis
- [ ] dbxNoiseMap wrapper and visualization
- [ ] dbxGhost wrapper and comparison
- [ ] dbxMetadata wrapper and extraction
- [ ] dbxHashFile wrapper and verification
- [ ] dbxCsvViewer wrapper and parsing
- [ ] dbxSeqCheck wrapper and validation
- [ ] dbxMouseRecorder wrapper and automation

### Phase 3: High-Level Integration (Days 5-6)
- [ ] ForensicsAnalyzer orchestrator
- [ ] EnhancedScreenshotIntelligence
- [ ] ImageForensics module
- [ ] MetadataExtractor module
- [ ] DataAnalyzer module

### Phase 4: API Integration (Day 7)
- [ ] Add 9 forensics capabilities to registry
- [ ] Implement capability handlers in unified API
- [ ] Natural language trigger configuration
- [ ] REST endpoint creation

### Phase 5: Workflows & Documentation (Days 8-9)
- [ ] Automated analysis pipeline
- [ ] Incident investigation workflow
- [ ] Evidence collection workflow
- [ ] Comprehensive documentation
- [ ] Usage examples and tutorials

### Phase 6: Testing & Validation (Day 10)
- [ ] Integration tests
- [ ] Performance benchmarking
- [ ] Documentation review
- [ ] Commit and deploy

---

## Security Considerations

### Tool Execution
- **Sandboxing**: Run Windows tools in isolated Wine environment
- **Resource Limits**: CPU and memory constraints
- **Timeout Protection**: 5-minute maximum execution
- **Input Validation**: Sanitize file paths and arguments

### Evidence Integrity
- **Hash Chains**: SHA-256 minimum for all evidence
- **Immutable Storage**: Write-once evidence directories
- **Audit Logging**: All forensic operations logged
- **Access Control**: Restrict modification permissions

### Privacy
- **Metadata Scrubbing**: Option to remove PII from reports
- **Encrypted Storage**: AES-256 for sensitive artifacts
- **Access Logs**: Track who accessed what evidence

---

## Success Metrics

### Coverage
- [ ] All 9 DBXForensics tools integrated
- [ ] 9 new capabilities registered
- [ ] 3 automated workflows operational

### Functionality
- [ ] Screenshot authenticity verification
- [ ] Automated forensic analysis pipeline
- [ ] Natural language forensics queries
- [ ] Integration with existing screenshot intelligence

### Performance
- [ ] <2s average tool execution time
- [ ] <5MB memory overhead per tool
- [ ] Batch analysis: 100+ screenshots/hour

### Quality
- [ ] 100% test coverage for wrappers
- [ ] Comprehensive documentation
- [ ] Integration tests passing
- [ ] No regression in existing features

---

## Next Steps

1. **Create forensics directory structure**
2. **Download DBXForensics tools** (Windows executables)
3. **Test Wine compatibility** on Linux
4. **Implement base DBXForensicsTool wrapper**
5. **Begin Phase 1 implementation**

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-17
**Author**: LAT5150 DRVMIL Integration Team
