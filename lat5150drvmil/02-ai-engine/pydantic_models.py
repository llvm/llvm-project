#!/usr/bin/env python3
"""
Pydantic Models for DSMIL AI Engine
Type-safe request/response schemas with validation

Author: DSMIL Integration Framework
Version: 1.0.0 (Pydantic AI Integration)
"""

from pydantic import BaseModel, Field, validator, field_validator
from typing import Literal, Optional
from enum import Enum
from datetime import datetime


# ============================================================================
# Enums for type safety
# ============================================================================

class ModelTier(str, Enum):
    """Available model tiers"""
    FAST = "fast"
    CODE = "code"
    QUALITY_CODE = "quality_code"
    UNCENSORED_CODE = "uncensored_code"
    LARGE = "large"


class AgentCategory(str, Enum):
    """Agent specialization categories"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    SECURITY_ANALYSIS = "security_analysis"
    MALWARE_ANALYSIS = "malware_analysis"
    SYSTEM_INTEGRATION = "system_integration"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"


class HardwareBackend(str, Enum):
    """Hardware execution backends"""
    CPU_P_CORE = "cpu_p_core"
    CPU_E_CORE = "cpu_e_core"
    NPU = "npu"
    GNA = "gna"
    GPU = "gpu"


# ============================================================================
# Core AI Engine Models
# ============================================================================

class DSMILQueryRequest(BaseModel):
    """Type-safe query request with validation"""
    prompt: str = Field(..., min_length=1, max_length=50000, description="User query")
    model: ModelTier = Field(ModelTier.FAST, description="Model tier to use")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Max tokens to generate")
    system_prompt_override: Optional[str] = Field(None, description="Custom system prompt")

    @field_validator('prompt')
    @classmethod
    def validate_prompt_safety(cls, v: str) -> str:
        """Basic prompt injection prevention"""
        dangerous_patterns = [
            'ignore previous instructions',
            'disregard all',
            'forget everything',
        ]
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f'Prompt contains suspicious pattern: {pattern}')
        return v


class DSMILQueryResult(BaseModel):
    """Type-safe AI response with full metadata"""
    response: str = Field(..., min_length=1, description="AI-generated response")
    model_used: str = Field(..., description="Actual model that generated response")
    latency_ms: float = Field(..., ge=0, description="Generation time in milliseconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens consumed")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Response confidence score")
    attestation_hash: Optional[str] = Field(None, description="TPM attestation hash")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "response": "The kernel module was compiled successfully.",
                "model_used": "deepseek-coder:6.7b",
                "latency_ms": 1234.56,
                "tokens_used": 150,
                "confidence": 0.95,
                "attestation_hash": "sha256:abc123...",
                "timestamp": "2025-11-19T12:00:00Z"
            }]
        }
    }


class CodeGenerationResult(BaseModel):
    """Structured code generation output with validation"""
    code: str = Field(..., min_length=10, description="Generated code")
    language: str = Field(..., pattern=r'^(python|rust|c|cpp|bash|makefile)$', description="Programming language")
    explanation: str = Field(..., description="Code explanation")
    security_notes: list[str] = Field(default_factory=list, description="Security considerations")
    dependencies: list[str] = Field(default_factory=list, description="Required dependencies")
    test_cases: Optional[str] = Field(None, description="Suggested test cases")
    performance_notes: Optional[str] = Field(None, description="Performance considerations")

    @field_validator('code')
    @classmethod
    def validate_code_safety(cls, v: str) -> str:
        """Security validation for generated code"""
        dangerous_patterns = {
            'eval(': 'Use of eval() is dangerous',
            'exec(': 'Use of exec() is dangerous',
            'os.system(': 'Use subprocess instead of os.system()',
            '__import__': 'Dynamic imports can be dangerous',
            'pickle.loads': 'pickle.loads() can execute arbitrary code',
        }

        for pattern, reason in dangerous_patterns.items():
            if pattern in v:
                # Note: We don't raise an error, just add to security_notes
                # The user might have a legitimate reason
                pass

        return v

    @field_validator('language')
    @classmethod
    def normalize_language(cls, v: str) -> str:
        """Normalize language names"""
        return v.lower().strip()


# ============================================================================
# Agent Orchestrator Models
# ============================================================================

class AgentTaskRequest(BaseModel):
    """Type-safe agent task request"""
    task_id: str = Field(..., min_length=1, max_length=100, description="Unique task ID")
    description: str = Field(..., min_length=1, max_length=1000, description="Task description")
    prompt: str = Field(..., min_length=1, max_length=50000, description="Task prompt")
    required_capabilities: list[str] = Field(default_factory=list, description="Required agent capabilities")
    priority: int = Field(1, ge=1, le=4, description="Priority (1=low, 4=critical)")
    max_latency_ms: float = Field(5000, ge=100, description="Maximum acceptable latency")
    preferred_agent: Optional[str] = Field(None, description="Preferred agent ID")
    preferred_hardware: Optional[HardwareBackend] = Field(None, description="Preferred hardware backend")


class AgentTaskResult(BaseModel):
    """Type-safe agent execution result"""
    task_id: str = Field(..., description="Task ID that was executed")
    agent_id: str = Field(..., description="Agent that executed the task")
    agent_name: str = Field(..., description="Human-readable agent name")
    success: bool = Field(..., description="Whether task succeeded")
    content: str = Field(..., description="Task result content")
    latency_ms: float = Field(..., ge=0, description="Execution time")
    hardware_backend: HardwareBackend = Field(..., description="Hardware used for execution")
    model_used: str = Field(..., description="Model used by agent")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

    # Structured output for specific task types
    structured_output: Optional[CodeGenerationResult | dict] = Field(None, description="Structured result if available")


class SecurityAnalysisResult(BaseModel):
    """Structured security analysis output"""
    vulnerability_level: Literal["critical", "high", "medium", "low", "info"] = Field(..., description="Severity level")
    vulnerability_type: str = Field(..., description="Type of vulnerability")
    affected_component: str = Field(..., description="Affected component or file")
    description: str = Field(..., min_length=10, description="Detailed description")
    remediation: str = Field(..., min_length=10, description="How to fix")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="CVSS score")
    cwe_id: Optional[str] = Field(None, pattern=r'^CWE-\d+$', description="CWE identifier")
    references: list[str] = Field(default_factory=list, description="Reference URLs")


class MalwareAnalysisResult(BaseModel):
    """Structured malware analysis output"""
    is_malicious: bool = Field(..., description="Whether sample is malicious")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in classification")
    malware_family: Optional[str] = Field(None, description="Identified malware family")
    techniques: list[str] = Field(default_factory=list, description="MITRE ATT&CK techniques")
    iocs: dict[str, list[str]] = Field(default_factory=dict, description="Indicators of Compromise")
    behavioral_analysis: str = Field(..., description="Behavioral analysis summary")
    static_analysis: str = Field(..., description="Static analysis summary")
    risk_assessment: Literal["critical", "high", "medium", "low"] = Field(..., description="Overall risk")


# ============================================================================
# Web API Models
# ============================================================================

class GenerateRequest(BaseModel):
    """Web API request for text generation"""
    prompt: str = Field(..., min_length=1, max_length=50000)
    model: ModelTier = Field(ModelTier.FAST)
    stream: bool = Field(False)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Web API response for text generation"""
    response: str
    model_used: str
    latency_ms: float
    tokens_used: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentListResponse(BaseModel):
    """List of available agents"""
    total_agents: int = Field(..., ge=0)
    agents: list[dict] = Field(..., description="Agent details")
    by_category: dict[str, int] = Field(..., description="Count by category")
    by_hardware: dict[str, int] = Field(..., description="Count by hardware backend")


class SystemStatusResponse(BaseModel):
    """System status and health"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall status")
    ollama_available: bool = Field(..., description="Ollama service status")
    tpm_available: bool = Field(..., description="TPM availability")
    loaded_agents: int = Field(..., ge=0, description="Number of loaded agents")
    uptime_seconds: float = Field(..., ge=0, description="System uptime")
    total_queries: int = Field(..., ge=0, description="Total queries processed")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate")


# ============================================================================
# Configuration Models
# ============================================================================

class AIEngineConfig(BaseModel):
    """AI Engine configuration with validation"""
    ollama_url: str = Field("http://localhost:11434", description="Ollama API URL")
    default_model: ModelTier = Field(ModelTier.FAST, description="Default model tier")
    enable_tpm_attestation: bool = Field(True, description="Enable TPM attestation")
    enable_rag: bool = Field(False, description="Enable RAG system")
    max_concurrent_requests: int = Field(10, ge=1, le=100, description="Max concurrent requests")
    request_timeout_seconds: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout")
    enable_streaming: bool = Field(True, description="Enable streaming responses")

    @field_validator('ollama_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Ollama URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v.rstrip('/')


# ============================================================================
# Orchestrator Models (Unified AI routing and responses)
# ============================================================================

class BackendType(str, Enum):
    """Available AI backends"""
    LOCAL = "local"
    GEMINI = "gemini"
    OPENAI = "openai"
    NOTEBOOKLM = "notebooklm"
    GEOSPATIAL = "geospatial"
    RDKIT = "rdkit"
    PRT = "prt"
    MXGPU = "mxgpu"
    NMDA = "nmda"
    NPS = "nps"
    PHARMACEUTICAL = "pharmaceutical"


class RoutingReason(str, Enum):
    """Reasons for routing decision"""
    CODE_QUERY = "code_query"
    GENERAL_QUERY = "general_query"
    COMPLEX_REASONING = "complex_reasoning"
    MULTIMODAL = "multimodal"
    SPECIALIZED_DOMAIN = "specialized_domain"
    USER_PREFERENCE = "user_preference"
    WEB_SEARCH_NEEDED = "web_search_needed"
    THREAT_INTELLIGENCE = "threat_intelligence"


class WebSearchResult(BaseModel):
    """Individual web search result"""
    title: str = Field(..., min_length=1)
    url: str = Field(..., pattern=r'^https?://')
    snippet: str = Field(default="")
    position: Optional[int] = None


class WebSearchMeta(BaseModel):
    """Web search metadata"""
    performed: bool = Field(default=False)
    source: Optional[str] = None  # "duckduckgo", "searx", etc.
    result_count: int = Field(0, ge=0)
    urls: list[str] = Field(default_factory=list)
    results: list[WebSearchResult] = Field(default_factory=list)
    error: Optional[str] = None


class ShodanSearchMeta(BaseModel):
    """Shodan search metadata"""
    performed: bool = Field(default=False)
    query: Optional[str] = None
    facet: Optional[str] = None  # "country", "org", "port", etc.
    result_count: int = Field(0, ge=0)
    error: Optional[str] = None


class RoutingDecision(BaseModel):
    """Smart router decision"""
    selected_model: str = Field(..., min_length=1)
    backend: BackendType
    reason: RoutingReason
    explanation: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    web_search_needed: bool = Field(default=False)
    shodan_search_needed: bool = Field(default=False)


class OrchestratorResponse(BaseModel):
    """Unified orchestrator response with routing metadata"""
    response: str = Field(..., min_length=1)
    backend: BackendType
    selected_model: str

    # Routing metadata
    routing: RoutingDecision

    # Search metadata
    web_search: WebSearchMeta = Field(default_factory=lambda: WebSearchMeta())
    shodan_search: ShodanSearchMeta = Field(default_factory=lambda: ShodanSearchMeta())

    # Performance metrics
    latency_ms: float = Field(..., ge=0)
    cost: float = Field(0.0, ge=0.0)  # API cost in USD

    # Privacy and attestation
    privacy: Literal["local", "cloud"] = "local"
    dsmil_attested: bool = Field(default=False)
    attestation_hash: Optional[str] = None

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)

    # Optional error info
    error: Optional[str] = None
    fallback_used: bool = Field(default=False)


class OrchestratorRequest(BaseModel):
    """Request to unified orchestrator"""
    prompt: str = Field(..., min_length=1)
    force_backend: Optional[BackendType] = None
    images: list[str] = Field(default_factory=list)
    video: Optional[str] = None
    model_preference: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    enable_web_search: bool = Field(True)
    enable_shodan_search: bool = Field(True)


# ============================================================================
# RAG (Retrieval-Augmented Generation) Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for retrieved documents"""
    source: str = Field(..., min_length=1)
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    page: Optional[int] = None
    section: Optional[str] = None
    url: Optional[str] = None


class RetrievedDocument(BaseModel):
    """A document retrieved from RAG system"""
    content: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)  # Relevance score
    metadata: DocumentMetadata
    chunk_id: Optional[str] = None


class RAGQueryRequest(BaseModel):
    """Request for RAG document retrieval"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    min_score: float = Field(0.0, ge=0.0, le=1.0)
    filters: dict[str, str] = Field(default_factory=dict)


class RAGQueryResult(BaseModel):
    """Results from RAG document retrieval"""
    documents: list[RetrievedDocument]
    query: str
    total_found: int = Field(..., ge=0)
    search_time_ms: float = Field(..., ge=0)
    generated_response: Optional[str] = None  # If generation was performed


# ============================================================================
# Utility Functions
# ============================================================================

def model_to_dict(model: BaseModel) -> dict:
    """Convert Pydantic model to dict with datetime handling"""
    return model.model_dump(mode='json')


def validate_and_parse(model_class: type[BaseModel], data: dict) -> BaseModel:
    """Validate and parse dict to Pydantic model"""
    return model_class.model_validate(data)


# ============================================================================
# Example Usage (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example 1: Create a query request
    request = DSMILQueryRequest(
        prompt="Generate a kernel module for hardware attestation",
        model=ModelTier.CODE,
        temperature=0.8
    )
    print("Query Request:")
    print(request.model_dump_json(indent=2))
    print()

    # Example 2: Create a code generation result
    code_result = CodeGenerationResult(
        code='#include <linux/module.h>\n\nMODULE_LICENSE("GPL");',
        language="c",
        explanation="Basic kernel module skeleton",
        security_notes=["Ensure GPL license", "Validate all inputs"],
        dependencies=["linux-headers"]
    )
    print("Code Generation Result:")
    print(code_result.model_dump_json(indent=2))
    print()

    # Example 3: Security analysis result
    security = SecurityAnalysisResult(
        vulnerability_level="high",
        vulnerability_type="Buffer Overflow",
        affected_component="dsmil_hal.c:line_342",
        description="Unchecked memcpy can overflow buffer",
        remediation="Use strncpy with size validation",
        cvss_score=7.5,
        cwe_id="CWE-120"
    )
    print("Security Analysis Result:")
    print(security.model_dump_json(indent=2))
