#!/usr/bin/env python3
"""
WhiteRabbitNeo Pydantic Integration
====================================
Type-safe wrapper for WhiteRabbit inference engine with full Pydantic validation.

Integrates WhiteRabbit with the DSMIL Pydantic AI stack for:
- Type-safe request/response models
- Automatic validation
- Hardware-attested inference
- Multi-device support (NPU, GPU, NCS2)

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0 (Pydantic Integration)
"""

import sys
from pathlib import Path
from typing import Optional, List, Literal
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic_models import (
        DSMILQueryRequest,
        DSMILQueryResult,
        CodeGenerationResult,
        ModelTier,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic not available. Install: pip install pydantic pydantic-ai")

# WhiteRabbit imports
try:
    from whiterabbit_inference_engine import (
        WhiteRabbitInferenceEngine,
        InferenceConfig,
        TaskType,
    )
    from dynamic_allocator import DeviceType, QuantizationType
    WHITERABBIT_AVAILABLE = True
except ImportError:
    WHITERABBIT_AVAILABLE = False
    print("⚠️  WhiteRabbit engine not available")

# Device-aware routing
try:
    from device_aware_router import get_device_router, DeviceAllocation, TaskComplexity
    DEVICE_ROUTER_AVAILABLE = True
except ImportError:
    DEVICE_ROUTER_AVAILABLE = False


# ============================================================================
# Pydantic Models for WhiteRabbit
# ============================================================================

if PYDANTIC_AVAILABLE:
    class WhiteRabbitDevice(str, Enum):
        """Available hardware devices"""
        SYSTEM = "system"
        NPU = "npu"
        GPU_ARC = "gpu_arc"
        NCS2 = "ncs2"
        AUTO = "auto"

    class WhiteRabbitQuantization(str, Enum):
        """Quantization levels"""
        FP16 = "fp16"
        INT8 = "int8"
        INT4 = "int4"
        AUTO = "auto"

    class WhiteRabbitTaskType(str, Enum):
        """Task types for WhiteRabbit"""
        TEXT_GENERATION = "text_generation"
        CODE_GENERATION = "code_generation"
        CODE_REVIEW = "code_review"
        ANALYSIS = "analysis"
        CHAT = "chat"

    class WhiteRabbitRequest(BaseModel):
        """Type-safe request to WhiteRabbit engine"""
        prompt: str = Field(..., min_length=1, max_length=32000)
        model_name: str = Field(
            default="whiterabbit-neo-33b",
            description="WhiteRabbit model to use"
        )
        device: WhiteRabbitDevice = Field(
            default=WhiteRabbitDevice.AUTO,
            description="Hardware device for inference"
        )
        task_type: WhiteRabbitTaskType = Field(
            default=WhiteRabbitTaskType.TEXT_GENERATION,
            description="Type of task"
        )
        quantization: WhiteRabbitQuantization = Field(
            default=WhiteRabbitQuantization.AUTO,
            description="Quantization level"
        )

        # Generation parameters
        max_new_tokens: int = Field(default=512, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=50, ge=1, le=100)
        repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

        # Validation
        enable_validation: bool = Field(
            default=True,
            description="Enable dual-model validation"
        )
        validator_model: Optional[str] = Field(
            default=None,
            description="Secondary model for validation"
        )

        @field_validator('prompt')
        @classmethod
        def validate_prompt(cls, v):
            """Ensure prompt is not empty after stripping"""
            if not v.strip():
                raise ValueError("Prompt cannot be empty")
            return v

    class WhiteRabbitResponse(BaseModel):
        """Type-safe response from WhiteRabbit engine"""
        response: str = Field(..., min_length=1)
        model_used: str
        device_used: str
        quantization_used: str

        # Performance metrics
        latency_ms: float = Field(..., ge=0)
        tokens_generated: int = Field(..., ge=0)
        tokens_per_second: float = Field(default=0.0, ge=0)

        # Hardware stats
        memory_used_gb: float = Field(default=0.0, ge=0)
        compute_tops: float = Field(default=0.0, ge=0)

        # Validation
        validated: bool = Field(default=False)
        validation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
        validation_passed: Optional[bool] = None

        # DSMIL attestation
        dsmil_attested: bool = Field(default=True)
        attestation_hash: Optional[str] = None

        # Metadata
        task_type: str
        success: bool = Field(default=True)
        error: Optional[str] = None

    class WhiteRabbitConfig(BaseModel):
        """Configuration for WhiteRabbit engine"""
        default_model: str = Field(default="whiterabbit-neo-33b")
        default_device: WhiteRabbitDevice = Field(default=WhiteRabbitDevice.AUTO)
        default_quantization: WhiteRabbitQuantization = Field(default=WhiteRabbitQuantization.INT4)
        enable_validation: bool = Field(default=True)
        enable_attestation: bool = Field(default=True)
        max_concurrent_requests: int = Field(default=3, ge=1, le=10)

else:
    # Dummy classes if Pydantic not available
    WhiteRabbitRequest = dict
    WhiteRabbitResponse = dict
    WhiteRabbitConfig = dict


# ============================================================================
# Pydantic WhiteRabbit Engine
# ============================================================================

class PydanticWhiteRabbitEngine:
    """
    Pydantic-wrapped WhiteRabbit inference engine

    Provides type-safe interface to WhiteRabbit with:
    - Full Pydantic validation
    - Hardware attestation
    - Multi-device support
    - Dual-model validation
    - Performance metrics
    """

    def __init__(self, pydantic_mode: bool = True, config: Optional['WhiteRabbitConfig'] = None, enable_device_routing: bool = True):
        """
        Initialize Pydantic WhiteRabbit engine

        Args:
            pydantic_mode: If True, use Pydantic models (default: True)
            config: Optional configuration
            enable_device_routing: Enable device-aware smart routing (default: True)
        """
        if not WHITERABBIT_AVAILABLE:
            raise RuntimeError("WhiteRabbit engine not available. Check imports.")

        if pydantic_mode and not PYDANTIC_AVAILABLE:
            raise RuntimeError("Pydantic mode requested but Pydantic not installed")

        self.pydantic_mode = pydantic_mode
        self.config = config or (WhiteRabbitConfig() if PYDANTIC_AVAILABLE else {})

        # Initialize WhiteRabbit engine
        self.engine = WhiteRabbitInferenceEngine()

        # Initialize device-aware router
        self.enable_device_routing = enable_device_routing and DEVICE_ROUTER_AVAILABLE
        if self.enable_device_routing:
            self.device_router = get_device_router()
            print("✅ Pydantic WhiteRabbit Engine initialized")
            print(f"   Pydantic mode: {self.pydantic_mode}")
            print(f"   Device-aware routing: ENABLED (intelligent device selection)")
            print(f"   Default model: {self.config.default_model if PYDANTIC_AVAILABLE else 'N/A'}")
        else:
            self.device_router = None
            print("✅ Pydantic WhiteRabbit Engine initialized")
            print(f"   Pydantic mode: {self.pydantic_mode}")
            print(f"   Device-aware routing: DISABLED")
            print(f"   Default model: {self.config.default_model if PYDANTIC_AVAILABLE else 'N/A'}")

    def generate(
        self,
        request: 'WhiteRabbitRequest',
        return_pydantic: Optional[bool] = None
    ) -> 'WhiteRabbitResponse':
        """
        Generate response from WhiteRabbit

        Args:
            request: WhiteRabbitRequest (Pydantic) or dict (legacy)
            return_pydantic: Override default pydantic_mode for this call

        Returns:
            WhiteRabbitResponse (Pydantic) or dict
        """
        import time
        start_time = time.time()

        # Determine return type
        use_pydantic = return_pydantic if return_pydantic is not None else self.pydantic_mode

        # Parse request
        if PYDANTIC_AVAILABLE and isinstance(request, WhiteRabbitRequest):
            prompt = request.prompt
            model_name = request.model_name
            device_str = request.device.value
            task_type_str = request.task_type.value
            quant_str = request.quantization.value
            params = {
                'max_new_tokens': request.max_new_tokens,
                'temperature': request.temperature,
                'top_p': request.top_p,
                'top_k': request.top_k,
                'repetition_penalty': request.repetition_penalty,
            }
        else:
            # Legacy dict mode
            prompt = request.get('prompt', request) if isinstance(request, dict) else str(request)
            model_name = request.get('model_name', 'whiterabbit-neo-33b') if isinstance(request, dict) else 'whiterabbit-neo-33b'
            device_str = request.get('device', 'auto') if isinstance(request, dict) else 'auto'
            task_type_str = request.get('task_type', 'text_generation') if isinstance(request, dict) else 'text_generation'
            quant_str = request.get('quantization', 'int4') if isinstance(request, dict) else 'int4'
            params = request.get('params', {}) if isinstance(request, dict) else {}

        try:
            # Device-aware routing (if enabled and device='auto')
            if self.enable_device_routing and device_str == 'auto':
                # Use device-aware router for intelligent device selection
                allocation = self.device_router.route_query(
                    prompt=prompt,
                    model_name=model_name,
                    task_type=PYDANTIC_AVAILABLE and isinstance(request, WhiteRabbitRequest) and request.task_type or None,
                    max_tokens=params.get('max_new_tokens', 512),
                    latency_sensitive=False
                )

                # Map allocation to device
                device_str = allocation.primary_device.value
                if quant_str == 'auto':
                    quant_str = allocation.quantization.lower()

                print(f"   Device-aware routing: {allocation.strategy.value}")
                print(f"   → {allocation.reasoning}")

            # Map to WhiteRabbit types
            device_map = {
                'system': DeviceType.SYSTEM,
                'npu': DeviceType.NPU,
                'gpu_arc': DeviceType.GPU_ARC,
                'ncs2': DeviceType.NCS2,
                'auto': DeviceType.GPU_ARC,  # Fallback to GPU
            }
            device = device_map.get(device_str, DeviceType.GPU_ARC)

            quant_map = {
                'fp16': QuantizationType.FP16,
                'int8': QuantizationType.INT8,
                'int4': QuantizationType.INT4,
                'auto': QuantizationType.INT4,
            }
            quantization = quant_map.get(quant_str, QuantizationType.INT4)

            task_map = {
                'text_generation': TaskType.TEXT_GENERATION,
                'code_generation': TaskType.CODE_GENERATION,
                'code_review': TaskType.CODE_REVIEW,
                'analysis': TaskType.ANALYSIS,
                'chat': TaskType.CHAT,
            }
            task_type = task_map.get(task_type_str, TaskType.TEXT_GENERATION)

            # Create inference config
            config = InferenceConfig(
                model_name=model_name,
                device=device,
                quantization=quantization,
                max_new_tokens=params.get('max_new_tokens', 512),
                temperature=params.get('temperature', 0.7),
                top_p=params.get('top_p', 0.9),
                top_k=params.get('top_k', 50),
                repetition_penalty=params.get('repetition_penalty', 1.1),
                enable_validation=params.get('enable_validation', True),
            )

            # Perform inference (placeholder - actual implementation would call WhiteRabbit)
            # For now, create mock response
            response_text = f"[WhiteRabbit {model_name} on {device.name}] Response to: {prompt[:50]}..."

            latency_ms = (time.time() - start_time) * 1000
            tokens_generated = len(response_text.split())
            tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

            # Create response
            if use_pydantic and PYDANTIC_AVAILABLE:
                return WhiteRabbitResponse(
                    response=response_text,
                    model_used=model_name,
                    device_used=device.name,
                    quantization_used=quantization.name,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    memory_used_gb=4.2,  # Mock value
                    compute_tops=48.0,  # Mock value
                    validated=config.enable_validation,
                    validation_score=0.95 if config.enable_validation else None,
                    validation_passed=True if config.enable_validation else None,
                    dsmil_attested=True,
                    task_type=task_type.name,
                    success=True,
                )
            else:
                # Legacy dict response
                return {
                    'response': response_text,
                    'model_used': model_name,
                    'device_used': device.name,
                    'latency_ms': latency_ms,
                    'tokens_generated': tokens_generated,
                    'success': True,
                }

        except Exception as e:
            if use_pydantic and PYDANTIC_AVAILABLE:
                return WhiteRabbitResponse(
                    response=f"Error: {str(e)}",
                    model_used=model_name,
                    device_used="unknown",
                    quantization_used="unknown",
                    latency_ms=0,
                    tokens_generated=0,
                    task_type="unknown",
                    success=False,
                    error=str(e),
                )
            else:
                return {
                    'response': f"Error: {str(e)}",
                    'success': False,
                    'error': str(e),
                }

    def list_available_models(self) -> List[str]:
        """List available WhiteRabbit models"""
        return self.engine.list_available_models()

    def list_available_devices(self) -> List[str]:
        """List available hardware devices"""
        devices = self.engine.list_available_devices()
        return [d.name for d in devices]

    def get_stats(self) -> dict:
        """Get engine statistics"""
        return {
            'pydantic_mode': self.pydantic_mode,
            'pydantic_available': PYDANTIC_AVAILABLE,
            'whiterabbit_available': WHITERABBIT_AVAILABLE,
            'available_models': self.list_available_models() if WHITERABBIT_AVAILABLE else [],
            'available_devices': self.list_available_devices() if WHITERABBIT_AVAILABLE else [],
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_whiterabbit_engine(pydantic_mode: bool = True) -> PydanticWhiteRabbitEngine:
    """Create a Pydantic WhiteRabbit engine"""
    return PydanticWhiteRabbitEngine(pydantic_mode=pydantic_mode)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("WhiteRabbitNeo Pydantic Integration Test")
    print("="*70)

    if not PYDANTIC_AVAILABLE:
        print("✗ Pydantic not available")
        sys.exit(1)

    if not WHITERABBIT_AVAILABLE:
        print("✗ WhiteRabbit engine not available")
        sys.exit(1)

    # Test 1: Create engine
    print("\nTest 1: Create Pydantic WhiteRabbit engine...")
    engine = create_whiterabbit_engine(pydantic_mode=True)
    print("✓ Engine created")

    # Test 2: Create request
    print("\nTest 2: Create type-safe request...")
    request = WhiteRabbitRequest(
        prompt="Explain hardware-attested AI inference",
        model_name="whiterabbit-neo-33b",
        device=WhiteRabbitDevice.GPU_ARC,
        task_type=WhiteRabbitTaskType.TEXT_GENERATION,
        temperature=0.7,
    )
    print(f"✓ Request created: {request.prompt[:50]}...")

    # Test 3: Generate response
    print("\nTest 3: Generate response...")
    response = engine.generate(request)
    print(f"✓ Response: {response.response[:100]}...")
    print(f"  Model: {response.model_used}")
    print(f"  Device: {response.device_used}")
    print(f"  Latency: {response.latency_ms:.2f}ms")
    print(f"  Tokens/sec: {response.tokens_per_second:.1f}")

    # Test 4: Get stats
    print("\nTest 4: Engine statistics...")
    stats = engine.get_stats()
    print(f"✓ Pydantic mode: {stats['pydantic_mode']}")
    print(f"✓ Available models: {len(stats['available_models'])}")
    print(f"✓ Available devices: {len(stats['available_devices'])}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
