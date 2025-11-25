#!/usr/bin/env python3
"""
Forensics Knowledge Integration for AI Engine

Provides forensics domain knowledge and capabilities to the Enhanced AI Engine:
- Forensic analysis terminology and concepts
- Tool capabilities and use cases
- Analysis workflow recommendations
- Natural language to forensic action mapping
- Best practices and methodologies

This module enables the AI to understand forensic concepts and provide
intelligent recommendations for evidence analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ForensicAnalysisType(Enum):
    """Types of forensic analysis"""
    AUTHENTICITY = "authenticity"  # ELA, manipulation detection
    DEVICE_ATTRIBUTION = "device_attribution"  # Noise pattern, fingerprinting
    METADATA_EXTRACTION = "metadata_extraction"  # EXIF, GPS, timestamps
    INTEGRITY_VERIFICATION = "integrity_verification"  # Hashing, chain of custody
    SEQUENCE_ANALYSIS = "sequence_analysis"  # Gap detection, completeness
    DATA_ANALYSIS = "data_analysis"  # CSV parsing, structured data
    VISUAL_COMPARISON = "visual_comparison"  # Screenshot comparison
    TIMELINE_RECONSTRUCTION = "timeline_reconstruction"  # Temporal ordering


class EvidenceType(Enum):
    """Types of digital evidence"""
    SCREENSHOT = "screenshot"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    LOG_FILE = "log_file"
    CSV_DATA = "csv_data"
    METADATA = "metadata"
    HASH_CHAIN = "hash_chain"


@dataclass
class ForensicConcept:
    """A forensic concept or technique"""
    name: str
    description: str
    use_cases: List[str]
    tools: List[str]
    reliability: str  # 'high', 'medium', 'low'
    court_admissible: bool
    requires_expert: bool


@dataclass
class ForensicWorkflow:
    """A complete forensic workflow"""
    name: str
    description: str
    evidence_types: List[str]
    steps: List[Dict]
    tools_required: List[str]
    estimated_duration: str
    output_format: str


class ForensicsKnowledge:
    """
    Forensics Knowledge Base

    Comprehensive knowledge of forensic analysis techniques, tools,
    and workflows for AI-assisted evidence analysis.
    """

    def __init__(self):
        """Initialize forensics knowledge base"""
        self.concepts = self._build_concepts()
        self.workflows = self._build_workflows()
        self.tool_mappings = self._build_tool_mappings()
        self.nl_to_action = self._build_nl_mappings()

    def _build_concepts(self) -> Dict[str, ForensicConcept]:
        """Build forensic concepts knowledge base"""
        return {
            'error_level_analysis': ForensicConcept(
                name="Error Level Analysis (ELA)",
                description="JPEG compression analysis technique that reveals areas of different compression levels, indicating potential manipulation. When an image is re-saved, manipulated areas show different error levels than the original portions.",
                use_cases=[
                    "Detect photoshopped regions in images",
                    "Identify composite images (multiple sources)",
                    "Verify screenshot authenticity",
                    "Detect cloned/copied regions",
                    "Find text or object insertion"
                ],
                tools=["dbxELA"],
                reliability="high",
                court_admissible=True,
                requires_expert=False
            ),

            'digital_noise_analysis': ForensicConcept(
                name="Digital Noise Pattern Analysis",
                description="Every digital camera/sensor produces a unique noise pattern (like a fingerprint) caused by sensor imperfections and processing algorithms. This pattern remains consistent across images from the same device.",
                use_cases=[
                    "Device identification and attribution",
                    "Verify image came from claimed source",
                    "Detect images from different cameras",
                    "Link screenshots to specific devices",
                    "Identify camera model/manufacturer"
                ],
                tools=["dbxNoiseMap"],
                reliability="high",
                court_admissible=True,
                requires_expert=True
            ),

            'metadata_forensics': ForensicConcept(
                name="Metadata Forensics",
                description="Analysis of embedded metadata (EXIF, XMP, IPTC) in digital files. Includes camera settings, GPS coordinates, timestamps, software used, and edit history.",
                use_cases=[
                    "Extract creation date and time",
                    "Obtain GPS location data",
                    "Identify camera model and settings",
                    "Detect editing software used",
                    "Recover deleted/hidden metadata"
                ],
                tools=["dbxMetadata"],
                reliability="high",
                court_admissible=True,
                requires_expert=False
            ),

            'cryptographic_hashing': ForensicConcept(
                name="Cryptographic Hash Functions",
                description="One-way mathematical functions that produce unique fixed-size fingerprints of files. Any change to the file produces a completely different hash, enabling integrity verification.",
                use_cases=[
                    "Verify file integrity over time",
                    "Create chain of custody",
                    "Detect tampered evidence",
                    "Prove file authenticity",
                    "Track evidence modifications"
                ],
                tools=["dbxHashFile"],
                reliability="absolute",
                court_admissible=True,
                requires_expert=False
            ),

            'sequence_integrity': ForensicConcept(
                name="Sequence Integrity Analysis",
                description="Verification that a series of numbered items (screenshots, log entries, transactions) is complete with no missing or duplicate items.",
                use_cases=[
                    "Detect missing screenshots in timeline",
                    "Verify log completeness",
                    "Identify gaps in evidence collection",
                    "Detect duplicate entries",
                    "Ensure ordered sequence correctness"
                ],
                tools=["dbxSeqCheck"],
                reliability="high",
                court_admissible=True,
                requires_expert=False
            ),

            'visual_comparison': ForensicConcept(
                name="Visual Screenshot Comparison",
                description="Side-by-side or overlay comparison of screenshots to detect changes, verify consistency, or identify alterations over time.",
                use_cases=[
                    "Compare before/after screenshots",
                    "Detect UI changes over time",
                    "Verify screenshot consistency",
                    "Identify subtle modifications",
                    "Track visual changes in evidence"
                ],
                tools=["dbxGhost"],
                reliability="medium",
                court_admissible=False,
                requires_expert=False
            ),

            'chain_of_custody': ForensicConcept(
                name="Chain of Custody",
                description="Documented chronological record tracking evidence handling from collection through presentation. Includes who handled it, when, why, and any changes made.",
                use_cases=[
                    "Prove evidence authenticity in court",
                    "Track evidence handling",
                    "Document access history",
                    "Maintain integrity trail",
                    "Prevent evidence challenges"
                ],
                tools=["dbxHashFile", "dbxMetadata", "dbxScreenshot"],
                reliability="absolute",
                court_admissible=True,
                requires_expert=False
            ),

            'forensic_capture': ForensicConcept(
                name="Forensically Sound Evidence Capture",
                description="Evidence collection that preserves original state, generates cryptographic hashes, records comprehensive metadata, and creates audit trail. Ensures admissibility.",
                use_cases=[
                    "Collect court-admissible screenshots",
                    "Document system state with integrity",
                    "Create tamper-evident evidence",
                    "Generate forensic metadata",
                    "Establish provenance"
                ],
                tools=["dbxScreenshot"],
                reliability="high",
                court_admissible=True,
                requires_expert=False
            )
        }

    def _build_workflows(self) -> Dict[str, ForensicWorkflow]:
        """Build forensic workflow templates"""
        return {
            'full_screenshot_analysis': ForensicWorkflow(
                name="Complete Screenshot Forensic Analysis",
                description="Comprehensive analysis of a screenshot including authenticity, attribution, metadata, and integrity",
                evidence_types=["screenshot", "image"],
                steps=[
                    {"step": 1, "action": "authenticity_check", "tool": "dbxELA", "description": "Detect manipulation"},
                    {"step": 2, "action": "device_fingerprint", "tool": "dbxNoiseMap", "description": "Identify source device"},
                    {"step": 3, "action": "metadata_extraction", "tool": "dbxMetadata", "description": "Extract all metadata"},
                    {"step": 4, "action": "hash_calculation", "tool": "dbxHashFile", "description": "Generate integrity hashes"}
                ],
                tools_required=["dbxELA", "dbxNoiseMap", "dbxMetadata", "dbxHashFile"],
                estimated_duration="30-60 seconds",
                output_format="ForensicAnalysisReport"
            ),

            'incident_investigation': ForensicWorkflow(
                name="Incident Investigation Workflow",
                description="Complete investigation of an incident with multiple screenshots and evidence items",
                evidence_types=["screenshot", "image", "log_file", "csv_data"],
                steps=[
                    {"step": 1, "action": "integrity_verification", "tool": "dbxHashFile", "description": "Verify no tampering since collection"},
                    {"step": 2, "action": "sequence_check", "tool": "dbxSeqCheck", "description": "Verify no missing screenshots"},
                    {"step": 3, "action": "batch_authenticity", "tool": "dbxELA", "description": "Check all screenshots for manipulation"},
                    {"step": 4, "action": "timeline_reconstruction", "tool": "dbxMetadata", "description": "Build temporal sequence"},
                    {"step": 5, "action": "device_attribution", "tool": "dbxNoiseMap", "description": "Verify device sources"}
                ],
                tools_required=["dbxHashFile", "dbxSeqCheck", "dbxELA", "dbxMetadata", "dbxNoiseMap"],
                estimated_duration="5-15 minutes",
                output_format="IncidentInvestigationReport"
            ),

            'evidence_collection': ForensicWorkflow(
                name="Forensic Evidence Collection",
                description="Proper collection of digital evidence with forensic metadata and integrity",
                evidence_types=["screenshot"],
                steps=[
                    {"step": 1, "action": "forensic_capture", "tool": "dbxScreenshot", "description": "Capture with metadata"},
                    {"step": 2, "action": "immediate_hashing", "tool": "dbxHashFile", "description": "Generate hash chain"},
                    {"step": 3, "action": "metadata_documentation", "tool": "dbxMetadata", "description": "Document all metadata"},
                    {"step": 4, "action": "ingestion", "tool": "EnhancedScreenshotIntelligence", "description": "Ingest to database"}
                ],
                tools_required=["dbxScreenshot", "dbxHashFile", "dbxMetadata"],
                estimated_duration="10-20 seconds per screenshot",
                output_format="ChainOfCustodyEntry"
            ),

            'authenticity_verification': ForensicWorkflow(
                name="Batch Authenticity Verification",
                description="Verify authenticity of multiple screenshots or images",
                evidence_types=["screenshot", "image"],
                steps=[
                    {"step": 1, "action": "batch_ela_analysis", "tool": "dbxELA", "description": "Detect manipulation in all images"},
                    {"step": 2, "action": "noise_pattern_analysis", "tool": "dbxNoiseMap", "description": "Verify device consistency"},
                    {"step": 3, "action": "generate_report", "tool": "ForensicsAnalyzer", "description": "Aggregate results"}
                ],
                tools_required=["dbxELA", "dbxNoiseMap"],
                estimated_duration="20-40 seconds per image",
                output_format="BatchAnalysisReport"
            )
        }

    def _build_tool_mappings(self) -> Dict[str, Dict]:
        """Map tools to their capabilities and use cases"""
        return {
            'dbxScreenshot': {
                'category': 'acquisition',
                'primary_function': 'Forensic screenshot capture',
                'outputs': ['screenshot', 'MD5', 'SHA1', 'SHA256', 'UTC timestamp', 'metadata'],
                'input_required': 'screen region (optional)',
                'typical_use': 'Evidence collection with forensic integrity'
            },

            'dbxELA': {
                'category': 'analysis',
                'primary_function': 'Error Level Analysis - manipulation detection',
                'outputs': ['ELA visualization', 'manipulation score', 'suspicious regions'],
                'input_required': 'image file, quality threshold',
                'typical_use': 'Detect photoshopped/edited images'
            },

            'dbxNoiseMap': {
                'category': 'analysis',
                'primary_function': 'Digital noise pattern extraction - device fingerprinting',
                'outputs': ['noise pattern map', 'device signature', 'fingerprint'],
                'input_required': 'image file',
                'typical_use': 'Identify which camera/device took image'
            },

            'dbxMetadata': {
                'category': 'analysis',
                'primary_function': 'Comprehensive metadata extraction',
                'outputs': ['EXIF data', 'GPS coordinates', 'timestamps', 'camera settings', 'software info'],
                'input_required': 'any file',
                'typical_use': 'Extract all embedded metadata from files'
            },

            'dbxHashFile': {
                'category': 'verification',
                'primary_function': 'Multi-algorithm cryptographic hashing',
                'outputs': ['CRC32', 'MD5', 'SHA-1', 'SHA-256', 'SHA-512', 'SHA3-256'],
                'input_required': 'any file',
                'typical_use': 'Generate integrity hashes for chain of custody'
            },

            'dbxSeqCheck': {
                'category': 'verification',
                'primary_function': 'Numeric sequence integrity verification',
                'outputs': ['missing numbers', 'duplicates', 'sequence status'],
                'input_required': 'list of numbers',
                'typical_use': 'Detect gaps in screenshot timelines or log sequences'
            },

            'dbxCsvViewer': {
                'category': 'analysis',
                'primary_function': 'CSV file analysis and Excel export',
                'outputs': ['parsed CSV', 'Excel export', 'column statistics'],
                'input_required': 'CSV file, delimiter',
                'typical_use': 'Analyze chat logs, system logs, or structured data exports'
            },

            'dbxGhost': {
                'category': 'comparison',
                'primary_function': 'Visual screenshot overlay comparison',
                'outputs': ['overlay visualization', 'difference highlights'],
                'input_required': 'two images, transparency level',
                'typical_use': 'Compare before/after screenshots to detect changes'
            },

            'dbxMouseRecorder': {
                'category': 'automation',
                'primary_function': 'Workflow automation and task recording',
                'outputs': ['automation script', 'recorded actions'],
                'input_required': 'recording duration or workflow script',
                'typical_use': 'Automate repetitive forensic analysis tasks'
            }
        }

    def _build_nl_mappings(self) -> Dict[str, Tuple[str, str]]:
        """Map natural language queries to forensic actions

        Returns:
            Dict mapping NL phrase -> (analysis_type, recommended_tool)
        """
        return {
            # Authenticity queries
            'is this real': ('authenticity', 'dbxELA'),
            'is this authentic': ('authenticity', 'dbxELA'),
            'is this photoshopped': ('authenticity', 'dbxELA'),
            'has this been edited': ('authenticity', 'dbxELA'),
            'detect manipulation': ('authenticity', 'dbxELA'),
            'check for tampering': ('authenticity', 'dbxELA'),
            'verify authenticity': ('authenticity', 'dbxELA'),

            # Device attribution queries
            'which device': ('device_attribution', 'dbxNoiseMap'),
            'which camera': ('device_attribution', 'dbxNoiseMap'),
            'device fingerprint': ('device_attribution', 'dbxNoiseMap'),
            'identify source': ('device_attribution', 'dbxNoiseMap'),
            'where did this come from': ('device_attribution', 'dbxNoiseMap'),

            # Metadata queries
            'when was this taken': ('metadata_extraction', 'dbxMetadata'),
            'where was this taken': ('metadata_extraction', 'dbxMetadata'),
            'extract metadata': ('metadata_extraction', 'dbxMetadata'),
            'show exif': ('metadata_extraction', 'dbxMetadata'),
            'get location': ('metadata_extraction', 'dbxMetadata'),
            'camera settings': ('metadata_extraction', 'dbxMetadata'),

            # Integrity queries
            'calculate hash': ('integrity_verification', 'dbxHashFile'),
            'verify integrity': ('integrity_verification', 'dbxHashFile'),
            'has this changed': ('integrity_verification', 'dbxHashFile'),
            'generate hash': ('integrity_verification', 'dbxHashFile'),

            # Sequence queries
            'check sequence': ('sequence_analysis', 'dbxSeqCheck'),
            'missing screenshots': ('sequence_analysis', 'dbxSeqCheck'),
            'detect gaps': ('sequence_analysis', 'dbxSeqCheck'),
            'verify completeness': ('sequence_analysis', 'dbxSeqCheck'),

            # Comparison queries
            'compare screenshots': ('visual_comparison', 'dbxGhost'),
            'what changed': ('visual_comparison', 'dbxGhost'),
            'show differences': ('visual_comparison', 'dbxGhost'),

            # Data analysis queries
            'analyze csv': ('data_analysis', 'dbxCsvViewer'),
            'parse log': ('data_analysis', 'dbxCsvViewer'),
            'examine data': ('data_analysis', 'dbxCsvViewer')
        }

    def get_concept(self, concept_name: str) -> Optional[ForensicConcept]:
        """Get forensic concept by name"""
        return self.concepts.get(concept_name)

    def get_workflow(self, workflow_name: str) -> Optional[ForensicWorkflow]:
        """Get forensic workflow by name"""
        return self.workflows.get(workflow_name)

    def interpret_query(self, query: str) -> Optional[Tuple[str, str, List[str]]]:
        """
        Interpret natural language forensic query

        Args:
            query: Natural language query

        Returns:
            Tuple of (analysis_type, primary_tool, recommended_actions) or None
        """
        query_lower = query.lower()

        # Direct mapping match
        for phrase, (analysis_type, tool) in self.nl_to_action.items():
            if phrase in query_lower:
                # Get recommended workflow
                workflow = None
                if 'full' in query_lower or 'complete' in query_lower or 'comprehensive' in query_lower:
                    workflow = 'full_screenshot_analysis'
                elif 'incident' in query_lower or 'investigation' in query_lower:
                    workflow = 'incident_investigation'
                elif 'batch' in query_lower or 'multiple' in query_lower:
                    workflow = 'authenticity_verification'

                recommended_actions = []
                if workflow and workflow in self.workflows:
                    wf = self.workflows[workflow]
                    recommended_actions = [step['action'] for step in wf.steps]

                return (analysis_type, tool, recommended_actions)

        return None

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get detailed information about a forensic tool"""
        return self.tool_mappings.get(tool_name)

    def recommend_workflow(self, evidence_type: str, goals: List[str]) -> Optional[ForensicWorkflow]:
        """
        Recommend workflow based on evidence type and analysis goals

        Args:
            evidence_type: Type of evidence
            goals: List of analysis goals (e.g., ['authenticity', 'attribution'])

        Returns:
            Recommended ForensicWorkflow or None
        """
        # Match workflows by evidence type
        matching = [
            wf for wf in self.workflows.values()
            if evidence_type in wf.evidence_types
        ]

        if not matching:
            return None

        # If multiple matches, prefer comprehensive workflows
        if len(matching) > 1:
            for wf in matching:
                if 'full' in wf.name.lower() or 'complete' in wf.name.lower():
                    return wf

        return matching[0]

    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get summary of all forensic capabilities"""
        return {
            'analysis_types': [t.value for t in ForensicAnalysisType],
            'evidence_types': [t.value for t in EvidenceType],
            'concepts': list(self.concepts.keys()),
            'workflows': list(self.workflows.keys()),
            'tools': list(self.tool_mappings.keys())
        }


if __name__ == "__main__":
    import json

    print("=== Forensics Knowledge Base ===\n")

    kb = ForensicsKnowledge()

    # Show capabilities
    print("Forensic Capabilities:")
    print(json.dumps(kb.get_all_capabilities(), indent=2))

    # Test query interpretation
    print("\n\nQuery Interpretation Examples:")
    print("-" * 50)

    test_queries = [
        "Is this screenshot authentic?",
        "Which device captured this image?",
        "Extract all metadata from this file",
        "Run full forensic analysis",
        "Check for missing screenshots in sequence"
    ]

    for query in test_queries:
        result = kb.interpret_query(query)
        if result:
            analysis_type, tool, actions = result
            print(f"\nQuery: '{query}'")
            print(f"  Type: {analysis_type}")
            print(f"  Tool: {tool}")
            print(f"  Actions: {', '.join(actions) if actions else 'N/A'}")

    # Show workflow details
    print("\n\nWorkflow: Full Screenshot Analysis")
    print("-" * 50)

    wf = kb.get_workflow('full_screenshot_analysis')
    if wf:
        print(f"Name: {wf.name}")
        print(f"Description: {wf.description}")
        print(f"Duration: {wf.estimated_duration}")
        print("\nSteps:")
        for step in wf.steps:
            print(f"  {step['step']}. {step['action']} ({step['tool']})")
            print(f"     → {step['description']}")

    print("\n✓ Forensics Knowledge Base ready")
