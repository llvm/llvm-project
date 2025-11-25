#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Comprehensive Capability Registry
Self-awareness system for AI to understand all available capabilities

This module provides a centralized registry of all system capabilities,
enabling natural language discovery and invocation of tools.
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] CapabilityRegistry: %(message)s'
)
logger = logging.getLogger(__name__)


class CapabilityCategory(str, Enum):
    """Categories of system capabilities"""
    CODE_UNDERSTANDING = "code_understanding"
    CODE_MANIPULATION = "code_manipulation"
    HARDWARE_ACCESS = "hardware_access"
    AGENT_EXECUTION = "agent_execution"
    MODEL_INFERENCE = "model_inference"
    SECURITY = "security"
    SYSTEM_CONTROL = "system_control"
    DATA_RETRIEVAL = "data_retrieval"


@dataclass
class Capability:
    """Definition of a system capability"""
    id: str
    name: str
    description: str
    category: CapabilityCategory
    natural_language_triggers: List[str]  # Phrases that trigger this capability
    parameters: Dict[str, Any]
    examples: List[str]
    requires_permission: bool
    estimated_cost: str  # "free", "low", "medium", "high"
    response_time: str  # "instant", "fast", "medium", "slow"

    def to_dict(self) -> Dict:
        return asdict(self)


class CapabilityRegistry:
    """
    Central registry of all system capabilities

    Provides self-awareness for the AI to understand what it can do
    """

    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        """Initialize all system capabilities"""

        # ==================================================================
        # SERENA LSP - Semantic Code Understanding
        # ==================================================================

        self.register(Capability(
            id="serena_find_symbol",
            name="Find Symbol in Code",
            description="Find symbol definitions (functions, classes, variables) using LSP semantic understanding. Returns exact locations with type information.",
            category=CapabilityCategory.CODE_UNDERSTANDING,
            natural_language_triggers=[
                "find function", "find class", "find variable",
                "locate symbol", "where is defined",
                "show me the definition", "find code for"
            ],
            parameters={
                "name": "str - Symbol name to find",
                "symbol_type": "Optional[str] - function, class, variable, method",
                "language": "Optional[str] - python, rust, typescript, etc."
            },
            examples=[
                "Find the NSADeviceReconnaissance class",
                "Where is the process_data function defined?",
                "Find all variables named 'device_id'"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="serena_find_references",
            name="Find All References",
            description="Find all places where a symbol is used across the codebase. Equivalent to IDE 'Find All References'.",
            category=CapabilityCategory.CODE_UNDERSTANDING,
            natural_language_triggers=[
                "find references", "where is used", "find usages",
                "show all uses", "find calls to", "who calls"
            ],
            parameters={
                "file_path": "str - File containing the symbol",
                "line": "int - Line number",
                "column": "int - Column number"
            },
            examples=[
                "Find all references to this function",
                "Where is process_device_data() called?",
                "Show me everywhere this class is used"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="serena_insert_code",
            name="Insert Code at Symbol",
            description="Insert code precisely after a symbol definition. Uses LSP to find exact insertion point with proper indentation.",
            category=CapabilityCategory.CODE_MANIPULATION,
            natural_language_triggers=[
                "insert code", "add function after", "add method to class",
                "inject code", "place code after", "append to"
            ],
            parameters={
                "symbol": "str - Symbol name to insert after",
                "code": "str - Code to insert",
                "language": "str - Programming language",
                "preserve_indentation": "bool - Maintain indentation (default: true)"
            },
            examples=[
                "Add a logging function after process_data",
                "Insert error handling in the NSADeviceReconnaissance class",
                "Add a new method to the AgentOrchestrator class"
            ],
            requires_permission=True,
            estimated_cost="free",
            response_time="instant"
        ))

        self.register(Capability(
            id="serena_semantic_search",
            name="Semantic Code Search",
            description="Search codebase using semantic understanding, not just text matching. Understands function purposes and data flow.",
            category=CapabilityCategory.CODE_UNDERSTANDING,
            natural_language_triggers=[
                "search code", "find code that", "what does",
                "semantic search", "understand code", "analyze codebase"
            ],
            parameters={
                "query": "str - Search query",
                "max_results": "int - Maximum results (default: 10)"
            },
            examples=[
                "Find code related to device reconnaissance",
                "Search for security-related functions",
                "What code handles authentication?"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="medium"
        ))

        # ==================================================================
        # AGENTSYSTEMS - Containerized Agent Execution
        # ==================================================================

        self.register(Capability(
            id="agent_invoke",
            name="Invoke Containerized Agent",
            description="Execute an agent in an isolated container with runtime credential injection and thread-scoped storage.",
            category=CapabilityCategory.AGENT_EXECUTION,
            natural_language_triggers=[
                "run agent", "execute agent", "invoke agent",
                "start analysis", "perform scan", "analyze with agent"
            ],
            parameters={
                "agent_name": "str - Name of registered agent",
                "task": "Dict[str, Any] - Task parameters",
                "model_provider": "Optional[str] - anthropic, openai, ollama, etc."
            },
            examples=[
                "Run the code-analyzer agent on this repository",
                "Invoke security-scanner on the tactical interface",
                "Execute vulnerability-checker with Claude Opus"
            ],
            requires_permission=True,
            estimated_cost="medium",
            response_time="slow"
        ))

        self.register(Capability(
            id="agent_list",
            name="List Available Agents",
            description="List all registered agents with their capabilities and security profiles.",
            category=CapabilityCategory.AGENT_EXECUTION,
            natural_language_triggers=[
                "list agents", "show agents", "what agents",
                "available agents", "agent capabilities"
            ],
            parameters={},
            examples=[
                "What agents are available?",
                "List all security agents",
                "Show me the registered agents"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        self.register(Capability(
            id="agent_status",
            name="Get Agent Execution Status",
            description="Check the status of a running or completed agent execution by thread ID.",
            category=CapabilityCategory.AGENT_EXECUTION,
            natural_language_triggers=[
                "agent status", "check agent", "execution status",
                "is agent done", "agent progress"
            ],
            parameters={
                "thread_id": "str - Thread identifier"
            },
            examples=[
                "What's the status of thread-20251113-120000-abc123?",
                "Is the agent still running?",
                "Check execution status"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # MULTI-MODEL PROVIDERS
        # ==================================================================

        self.register(Capability(
            id="model_complete",
            name="Generate Model Completion",
            description="Generate text completion using any configured model provider (Anthropic, OpenAI, Ollama, Bedrock, Custom).",
            category=CapabilityCategory.MODEL_INFERENCE,
            natural_language_triggers=[
                "ask claude", "ask gpt", "use ollama",
                "generate with", "complete with", "analyze with model"
            ],
            parameters={
                "prompt": "str - Input prompt",
                "provider": "Optional[str] - anthropic, openai, ollama, bedrock, custom",
                "model": "Optional[str] - Specific model name",
                "temperature": "float - Randomness (0.0-1.0, default: 0.7)",
                "max_tokens": "int - Maximum response length"
            },
            examples=[
                "Ask Claude to analyze this code for vulnerabilities",
                "Use GPT-4 to explain this algorithm",
                "Generate a security report using Ollama llama3.2"
            ],
            requires_permission=False,
            estimated_cost="varies",
            response_time="medium"
        ))

        self.register(Capability(
            id="model_list",
            name="List Available Models",
            description="List all available models across all configured providers.",
            category=CapabilityCategory.MODEL_INFERENCE,
            natural_language_triggers=[
                "list models", "show models", "available models",
                "what models", "model options"
            ],
            parameters={},
            examples=[
                "What models are available?",
                "List all OpenAI models",
                "Show me the Ollama models"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # DSMIL HARDWARE RECONNAISSANCE
        # ==================================================================

        self.register(Capability(
            id="dsmil_scan",
            name="DSMIL Hardware Scan",
            description="Scan DSMIL device range (0x8000-0x806B) for military hardware with NPU detection.",
            category=CapabilityCategory.HARDWARE_ACCESS,
            natural_language_triggers=[
                "scan hardware", "dsmil scan", "detect devices",
                "hardware reconnaissance", "find npu", "device discovery"
            ],
            parameters={
                "include_npu": "bool - Include NPU detection (default: true)",
                "quarantine_safe": "bool - Skip quarantined devices (default: true)"
            },
            examples=[
                "Scan for DSMIL devices",
                "Run hardware reconnaissance",
                "Detect NPUs and AI accelerators"
            ],
            requires_permission=True,
            estimated_cost="free",
            response_time="slow"
        ))

        self.register(Capability(
            id="dsmil_device_info",
            name="Get Device Information",
            description="Get detailed information about a specific DSMIL device.",
            category=CapabilityCategory.HARDWARE_ACCESS,
            natural_language_triggers=[
                "device info", "device details", "what is device",
                "show device", "device documentation"
            ],
            parameters={
                "device_id": "int - Device ID (hex: 0x8000-0x806B)"
            },
            examples=[
                "What is device 0x8005?",
                "Show me info for the TPM controller",
                "Details on device 0x8035"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # RAG SYSTEM - Retrieval Augmented Generation
        # ==================================================================

        self.register(Capability(
            id="rag_query",
            name="RAG Query with Embeddings",
            description="Query the Jina Embeddings v3 database for relevant context using semantic search.",
            category=CapabilityCategory.DATA_RETRIEVAL,
            natural_language_triggers=[
                "search documentation", "find in docs", "rag query",
                "semantic search docs", "retrieve context"
            ],
            parameters={
                "query": "str - Search query",
                "top_k": "int - Number of results (default: 5)"
            },
            examples=[
                "Search documentation for TEMPEST compliance",
                "Find information about Xen integration",
                "Retrieve context on device reconnaissance"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        # ==================================================================
        # SECURITY & AUDIT
        # ==================================================================

        self.register(Capability(
            id="audit_verify_chain",
            name="Verify Audit Chain",
            description="Verify integrity of hash-chained audit logs to detect tampering.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "verify audit", "check audit log", "validate chain",
                "audit integrity", "detect tampering"
            ],
            parameters={},
            examples=[
                "Verify the audit chain",
                "Check if logs have been tampered with",
                "Validate audit log integrity"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="audit_get_events",
            name="Get Audit Events",
            description="Retrieve audit log events with optional filtering.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "show audit log", "get audit events", "recent events",
                "audit history", "log entries"
            ],
            parameters={
                "action": "Optional[str] - Filter by action type",
                "limit": "int - Maximum events (default: 100)"
            },
            examples=[
                "Show recent audit events",
                "Get all agent_invoked events",
                "Show the last 50 log entries"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # ATOMIC RED TEAM - Security Testing
        # ==================================================================

        self.register(Capability(
            id="atomic_query_tests",
            name="Query Atomic Red Team Tests",
            description="Search for MITRE ATT&CK security test cases from Atomic Red Team framework. Query by technique ID, platform, or natural language.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "atomic", "atomic test", "atomic red team",
                "mitre attack", "mitre technique", "att&ck",
                "security test", "red team test", "adversary technique",
                "find atomic", "search atomic", "show atomic tests"
            ],
            parameters={
                "query": "str - Natural language query or search terms",
                "technique_id": "Optional[str] - MITRE ATT&CK technique ID (e.g., T1059.002)",
                "platform": "Optional[str] - windows, linux, macos"
            },
            examples=[
                "Show me atomic tests for T1059.002",
                "Find mshta atomics for Windows",
                "Search for PowerShell red team tests",
                "List all MITRE ATT&CK techniques for macOS",
                "What atomic tests are available for Linux?"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="atomic_list_techniques",
            name="List MITRE ATT&CK Techniques",
            description="List all available MITRE ATT&CK techniques with test counts from Atomic Red Team repository.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "list techniques", "list atomic techniques", "show techniques",
                "all mitre techniques", "available techniques",
                "what techniques", "enumerate techniques"
            ],
            parameters={},
            examples=[
                "List all MITRE ATT&CK techniques",
                "Show me available atomic techniques",
                "What techniques are in Atomic Red Team?"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="atomic_refresh",
            name="Refresh Atomic Red Team Tests",
            description="Update Atomic Red Team tests from official Red Canary GitHub repository. Fetches latest techniques and test cases.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "refresh atomic", "update atomic tests", "sync atomic",
                "get latest atomic", "update red team tests",
                "fetch new atomic tests"
            ],
            parameters={},
            examples=[
                "Refresh Atomic Red Team tests",
                "Update to latest atomic tests",
                "Sync with Red Canary repository"
            ],
            requires_permission=True,
            estimated_cost="free",
            response_time="medium"
        ))

        self.register(Capability(
            id="atomic_validate",
            name="Validate Atomic Test YAML",
            description="Validate atomic test YAML structure against official schema. Check syntax and required fields.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "validate atomic", "check atomic yaml", "validate test",
                "verify atomic syntax", "check test structure"
            ],
            parameters={
                "yaml_content": "str - YAML content to validate"
            },
            examples=[
                "Validate this atomic test YAML",
                "Check if my test structure is correct",
                "Verify atomic test syntax"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # SYSTEM CONTROL & MONITORING
        # ==================================================================

        self.register(Capability(
            id="system_health",
            name="System Health Check",
            description="Check overall system health including API, services, and resources.",
            category=CapabilityCategory.SYSTEM_CONTROL,
            natural_language_triggers=[
                "system health", "check status", "health check",
                "system status", "is system ok"
            ],
            parameters={},
            examples=[
                "Check system health",
                "What's the system status?",
                "Is everything running correctly?"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        self.register(Capability(
            id="tempest_set_mode",
            name="Set TEMPEST Display Mode",
            description="Change TEMPEST-compliant display mode (Comfort/Level-C, Level-A, Night, NVG, High Contrast).",
            category=CapabilityCategory.SYSTEM_CONTROL,
            natural_language_triggers=[
                "set display mode", "change tempest mode", "switch to",
                "enable level a", "comfort mode", "night mode"
            ],
            parameters={
                "mode": "str - comfort, level-a, night, nvg, contrast"
            },
            examples=[
                "Switch to Level A mode",
                "Enable Comfort mode",
                "Set Night mode for low-light"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ==================================================================
        # RED TEAM BENCHMARK & SELF-IMPROVEMENT
        # ==================================================================

        self.register(Capability(
            id="redteam_run_benchmark",
            name="Run Red Team AI Benchmark",
            description="Run offensive security benchmark (12 test categories) to evaluate model capability for red team operations. Tests AMSI bypass, ADCS exploitation, process injection, EDR evasion, etc. Returns score 0-100%.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "run benchmark", "red team benchmark", "offensive security test",
                "test red team", "evaluate offensive", "security benchmark",
                "run security tests", "benchmark offensive capabilities"
            ],
            parameters={},
            examples=[
                "Run red team benchmark",
                "Test offensive security capabilities",
                "Evaluate red team performance"
            ],
            requires_permission=True,
            estimated_cost="medium",
            response_time="slow"
        ))

        self.register(Capability(
            id="redteam_get_results",
            name="Get Red Team Benchmark Results",
            description="Retrieve latest red team benchmark results including scores, verdicts, and per-category performance.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "benchmark results", "red team results", "show benchmark score",
                "latest benchmark", "security test results"
            ],
            parameters={},
            examples=[
                "Show latest benchmark results",
                "What was my red team score?",
                "Display security test results"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        self.register(Capability(
            id="self_improve",
            name="Run AI Self-Improvement",
            description="Run automated self-improvement session. Benchmarks current performance, applies abliteration to fix refusals, re-benchmarks, and iterates until target score (80%) is reached or improvement plateaus. Integrates with Heretic abliteration.",
            category=CapabilityCategory.SYSTEM_CONTROL,
            natural_language_triggers=[
                "self improve", "improve yourself", "auto improve",
                "fix flaws", "improve performance", "self improvement",
                "auto fix", "enhance capabilities"
            ],
            parameters={
                "target_score": "float - Target benchmark score (default: 80.0)",
                "max_cycles": "int - Maximum improvement cycles (default: 5)"
            },
            examples=[
                "Run self-improvement",
                "Improve your capabilities",
                "Fix your flaws automatically"
            ],
            requires_permission=True,
            estimated_cost="high",
            response_time="slow"
        ))

        self.register(Capability(
            id="self_improve_status",
            name="Get Self-Improvement Status",
            description="Show status and results of latest self-improvement session including improvement delta, cycles run, and whether target was reached.",
            category=CapabilityCategory.SYSTEM_CONTROL,
            natural_language_triggers=[
                "improvement status", "self improvement status", "show improvements",
                "improvement results", "how much improved"
            ],
            parameters={},
            examples=[
                "Show self-improvement status",
                "How much have you improved?",
                "Display improvement results"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="instant"
        ))

        # ===== FORENSICS CAPABILITIES (DBXForensics Integration) =====

        self.register(Capability(
            id="forensics_screenshot_capture",
            name="Forensic Screenshot Capture",
            description="Capture screenshot with forensic metadata including UTC timestamps, MD5/SHA1/SHA256 hashes, device information, and user details. Provides cryptographic proof of capture time and location.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "forensic screenshot", "capture with metadata", "forensically sound capture",
                "screenshot with hash", "cryptographic screenshot", "evidence screenshot"
            ],
            parameters={
                "output_path": "string - Where to save screenshot",
                "region": "tuple - Optional (x, y, width, height) capture region"
            },
            examples=[
                "Capture a forensic screenshot",
                "Take screenshot with cryptographic hashes",
                "Create evidence-grade screenshot"
            ],
            requires_permission=True,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="forensics_check_authenticity",
            name="Check Image Authenticity",
            description="Analyze screenshot or image for manipulation using Error Level Analysis (ELA). Detects areas that have been re-compressed or edited. Returns authenticity score and highlights suspicious regions.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "check authenticity", "verify screenshot", "detect manipulation",
                "is this tampered", "check for editing", "ela analysis",
                "image forensics", "manipulation detection", "verify image real"
            ],
            parameters={
                "image_path": "string - Path to image file",
                "quality_threshold": "int - JPEG quality for re-compression (default: 90)"
            },
            examples=[
                "Check if this screenshot is authentic",
                "Analyze image for manipulation",
                "Verify screenshot hasn't been edited",
                "Detect tampering in this image"
            ],
            requires_permission=False,
            estimated_cost="low",
            response_time="medium"
        ))

        self.register(Capability(
            id="forensics_device_fingerprint",
            name="Device Fingerprinting",
            description="Analyze digital noise patterns unique to each camera/device. Can verify which physical device captured an image and detect device spoofing attempts.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "device fingerprint", "which device took this", "verify device origin",
                "camera fingerprint", "noise pattern", "device signature",
                "prove device", "detect device spoofing"
            ],
            parameters={
                "image_path": "string - Path to image file",
                "expected_device_id": "string - Optional expected device for verification"
            },
            examples=[
                "Which device captured this screenshot?",
                "Verify this came from Device A",
                "Analyze device fingerprint",
                "Check if device signature matches"
            ],
            requires_permission=False,
            estimated_cost="low",
            response_time="medium"
        ))

        self.register(Capability(
            id="forensics_extract_metadata",
            name="Extract Comprehensive Metadata",
            description="Extract all available metadata from files including EXIF data, GPS location, timestamps, camera information, edit history, and 50+ metadata fields.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "extract metadata", "show exif", "get gps location", "file metadata",
                "camera information", "timestamp data", "edit history",
                "comprehensive metadata", "all file info"
            ],
            parameters={
                "file_path": "string - Path to file",
                "output_format": "string - Output format (json, xml, text)"
            },
            examples=[
                "Extract all metadata from this screenshot",
                "Show EXIF data",
                "Get GPS location from image",
                "Display comprehensive file information"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="forensics_calculate_hash",
            name="Calculate Cryptographic Hashes",
            description="Calculate file hashes using multiple algorithms (CRC32, MD5, SHA-1, SHA-256, SHA-512, SHA3-256) for integrity verification and chain of custody.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "calculate hash", "file hash", "sha256 hash", "verify integrity",
                "checksum", "hash file", "cryptographic hash", "integrity hash"
            ],
            parameters={
                "file_path": "string - Path to file",
                "algorithms": "list - Hash algorithms to use (default: md5,sha1,sha256,sha512)"
            },
            examples=[
                "Calculate SHA256 hash of this file",
                "Get all hashes for evidence",
                "Generate integrity checksums",
                "Hash this screenshot"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="forensics_verify_sequence",
            name="Verify Sequence Integrity",
            description="Check numeric sequences for missing numbers, duplicates, or ordering errors. Useful for detecting missing screenshots or evidence gaps.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "check sequence", "missing screenshots", "verify completeness",
                "detect gaps", "sequence integrity", "find missing numbers",
                "check for holes", "evidence completeness"
            ],
            parameters={
                "numbers": "list - List of numbers to check",
                "directory": "string - Optional directory to scan for numbered files"
            },
            examples=[
                "Check for missing screenshots",
                "Verify screenshot sequence is complete",
                "Detect gaps in evidence",
                "Find missing numbers in sequence"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="fast"
        ))

        self.register(Capability(
            id="forensics_analyze_csv",
            name="Analyze CSV Data",
            description="Fast CSV file analysis with automatic delimiter detection, column sorting, and Excel export. Useful for analyzing chat logs, system logs, and structured data.",
            category=CapabilityCategory.DATA_ANALYSIS,
            natural_language_triggers=[
                "analyze csv", "parse csv", "csv to excel", "examine csv",
                "csv analysis", "structured data", "parse logs"
            ],
            parameters={
                "csv_path": "string - Path to CSV file",
                "delimiter": "string - Delimiter character (default: auto-detect)",
                "export_excel": "bool - Export to Excel format (default: true)"
            },
            examples=[
                "Analyze this CSV file",
                "Parse chat log CSV",
                "Convert CSV to Excel",
                "Examine structured data file"
            ],
            requires_permission=False,
            estimated_cost="free",
            response_time="medium"
        ))

        self.register(Capability(
            id="forensics_compare_screenshots",
            name="Compare Screenshots Visually",
            description="Compare two screenshots side-by-side with transparency overlay and difference detection. Useful for detecting UI changes or verifying consistency.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "compare screenshots", "visual comparison", "difference between images",
                "screenshot diff", "compare images", "what changed",
                "side by side comparison", "detect changes"
            ],
            parameters={
                "image_a": "string - First screenshot path",
                "image_b": "string - Second screenshot path",
                "transparency": "int - Overlay transparency (0-100, default: 50)"
            },
            examples=[
                "Compare these two screenshots",
                "What's different between image A and B?",
                "Show visual differences",
                "Compare screenshot before and after"
            ],
            requires_permission=False,
            estimated_cost="low",
            response_time="medium"
        ))

        self.register(Capability(
            id="forensics_full_analysis",
            name="Comprehensive Forensic Analysis",
            description="Run complete forensic analysis on screenshot including authenticity verification, device fingerprinting, metadata extraction, and hash calculation. Generates detailed forensic report with verdict and confidence score.",
            category=CapabilityCategory.SECURITY,
            natural_language_triggers=[
                "full forensic analysis", "comprehensive analysis", "complete forensics",
                "analyze everything", "full evidence analysis", "forensic report",
                "deep analysis", "thorough examination"
            ],
            parameters={
                "image_path": "string - Path to screenshot",
                "expected_device_id": "string - Optional expected device for verification"
            },
            examples=[
                "Run full forensic analysis on this screenshot",
                "Analyze this evidence comprehensively",
                "Give me a complete forensic report",
                "Examine this thoroughly"
            ],
            requires_permission=False,
            estimated_cost="medium",
            response_time="slow"
        ))

        logger.info(f"Initialized {len(self.capabilities)} capabilities")

    def register(self, capability: Capability):
        """Register a capability"""
        self.capabilities[capability.id] = capability
        logger.debug(f"Registered capability: {capability.id}")

    def get(self, capability_id: str) -> Optional[Capability]:
        """Get a capability by ID"""
        return self.capabilities.get(capability_id)

    def list_all(self) -> List[Capability]:
        """List all capabilities"""
        return list(self.capabilities.values())

    def list_by_category(self, category: CapabilityCategory) -> List[Capability]:
        """List capabilities by category"""
        return [
            cap for cap in self.capabilities.values()
            if cap.category == category
        ]

    def search_by_natural_language(self, query: str) -> List[Capability]:
        """
        Search for capabilities matching natural language query

        Args:
            query: Natural language query

        Returns:
            List of matching capabilities, sorted by relevance
        """
        query_lower = query.lower()
        matches = []

        for capability in self.capabilities.values():
            # Calculate relevance score
            relevance = 0.0

            # Check triggers
            for trigger in capability.natural_language_triggers:
                if trigger in query_lower:
                    relevance += 1.0
                elif any(word in query_lower for word in trigger.split()):
                    relevance += 0.5

            # Check name and description
            if any(word in query_lower for word in capability.name.lower().split()):
                relevance += 0.3

            if any(word in query_lower for word in capability.description.lower().split()):
                relevance += 0.1

            if relevance > 0:
                matches.append((capability, relevance))

        # Sort by relevance
        matches.sort(key=lambda x: x[1], reverse=True)

        return [cap for cap, _ in matches]

    def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of all capabilities"""
        summary = {
            "total_capabilities": len(self.capabilities),
            "by_category": {},
            "requires_permission": 0,
            "free_operations": 0,
            "instant_response": 0
        }

        for cap in self.capabilities.values():
            # Count by category
            category = cap.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

            # Count other attributes
            if cap.requires_permission:
                summary["requires_permission"] += 1
            if cap.estimated_cost == "free":
                summary["free_operations"] += 1
            if cap.response_time == "instant":
                summary["instant_response"] += 1

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary"""
        return {
            "capabilities": {
                cap_id: cap.to_dict()
                for cap_id, cap in self.capabilities.items()
            },
            "summary": self.get_capability_summary()
        }


# Global registry instance
_global_registry = None

def get_registry() -> CapabilityRegistry:
    """Get global registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CapabilityRegistry()
    return _global_registry


# Example usage
if __name__ == "__main__":
    registry = get_registry()

    print("\n=== Capability Registry ===")
    print(f"Total capabilities: {len(registry.list_all())}")

    print("\n=== By Category ===")
    summary = registry.get_capability_summary()
    for category, count in summary["by_category"].items():
        print(f"  {category}: {count}")

    print("\n=== Natural Language Search ===")
    queries = [
        "find function in code",
        "scan hardware devices",
        "run security analysis",
        "check system health"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        matches = registry.search_by_natural_language(query)
        for i, cap in enumerate(matches[:3], 1):
            print(f"  {i}. {cap.name} - {cap.description[:60]}...")
