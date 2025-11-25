"""
WhiteRabbitNeo Unified Inference Engine
========================================
Complete inference system with:
- Multi-device support (System/NPU/GPU/NCS2)
- Runtime device switching
- Dynamic allocation adjustment
- Validation pipeline integration
- Interactive model selection

System: Dell Latitude 5450 - 64GB RAM, 48GB NCS2, 106GB total
Total Compute: 160-260 TOPS

Author: LAT5150DRVMIL AI Platform
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from dynamic_allocator import DeviceType, QuantizationType, get_allocator
from hardware_profile import get_hardware_profile
from validation_pipeline import ValidationMode, get_validation_pipeline
from whiterabbit_model_manager import (
    InferenceMode,
    ModelState,
    get_model_manager,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Inference task type."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    ANALYSIS = "analysis"
    CHAT = "chat"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Model selection
    model_name: str
    device: DeviceType = DeviceType.GPU_ARC
    inference_mode: InferenceMode = InferenceMode.AUTO
    quantization: QuantizationType = QuantizationType.INT4

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Validation
    enable_validation: bool = True
    validation_mode: ValidationMode = ValidationMode.DUAL
    validator_model: Optional[str] = None  # Secondary model for validation

    # Performance
    batch_size: int = 1
    use_cache: bool = True


class WhiteRabbitInferenceEngine:
    """
    Unified inference engine for WhiteRabbitNeo models.

    Coordinates:
    - Model loading and management
    - Device allocation and switching
    - Validation pipeline
    - Multi-model workflows
    """

    def __init__(self):
        """Initialize inference engine."""
        self.model_manager = get_model_manager()
        self.allocator = get_allocator()
        self.validator = get_validation_pipeline()
        self.hardware = get_hardware_profile()

        self.active_config: Optional[InferenceConfig] = None

        logger.info("WhiteRabbitNeo Inference Engine initialized")
        logger.info(f"  Total memory: {self.hardware.get_total_memory_gb():.0f}GB")
        logger.info(f"  Total TOPS: {self.hardware.total_system_tops:.0f}")

    def list_available_models(self) -> List[str]:
        """List available models."""
        return self.model_manager.list_models()

    def list_available_devices(self) -> List[DeviceType]:
        """List available devices."""
        devices = []

        if self.hardware.arc_gpu_available:
            devices.append(DeviceType.GPU_ARC)
        if self.hardware.npu_available:
            devices.append(DeviceType.NPU)
        if self.hardware.ncs2_available:
            devices.append(DeviceType.NCS2)
        devices.append(DeviceType.CPU)  # Always available

        return devices

    def setup(
        self,
        model_name: str,
        device: DeviceType = DeviceType.GPU_ARC,
        quantization: QuantizationType = QuantizationType.INT4,
        validator_model: Optional[str] = None
    ) -> bool:
        """
        Setup inference engine with specified model.

        Args:
            model_name: Model to load
            device: Primary device
            quantization: Quantization type
            validator_model: Optional validator model

        Returns:
            True if successful
        """
        logger.info(f"Setting up inference engine:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device.value}")
        logger.info(f"  Quantization: {quantization.value}")

        # Load primary model
        success = self.model_manager.load_model(
            model_name,
            quantization=quantization,
            inference_mode=InferenceMode.AUTO
        )

        if not success:
            logger.error("Failed to load primary model")
            return False

        # Load validator model if specified
        if validator_model and validator_model != model_name:
            logger.info(f"Loading validator model: {validator_model}")
            success = self.model_manager.load_model(
                validator_model,
                quantization=QuantizationType.INT8,  # Lighter quantization for validator
                inference_mode=InferenceMode.AUTO
            )

            if not success:
                logger.warning("Failed to load validator model")

        # Create default config
        self.active_config = InferenceConfig(
            model_name=model_name,
            device=device,
            quantization=quantization,
            validator_model=validator_model
        )

        logger.info("✓ Setup complete")
        return True

    def switch_device(self, target_device: DeviceType) -> bool:
        """
        Switch active model to different device.

        Args:
            target_device: Target device

        Returns:
            True if successful
        """
        if not self.active_config:
            logger.error("No active configuration")
            return False

        model_name = self.active_config.model_name

        logger.info(f"Switching {model_name} to {target_device.value}...")

        success = self.model_manager.switch_device(
            model_name,
            target_device
        )

        if success:
            self.active_config.device = target_device
            logger.info("✓ Device switched")

        return success

    def update_config(self, **kwargs):
        """
        Update inference configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        if not self.active_config:
            logger.error("No active configuration")
            return

        for key, value in kwargs.items():
            if hasattr(self.active_config, key):
                setattr(self.active_config, key, value)
                logger.info(f"Updated {key} = {value}")

    def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.TEXT_GENERATION,
        validate: bool = True,
        **kwargs
    ) -> Tuple[str, Optional[object]]:
        """
        Generate text with optional validation.

        Args:
            prompt: Input prompt
            task_type: Task type
            validate: Enable validation
            **kwargs: Override config parameters

        Returns:
            (generated_text, validation_report)
        """
        if not self.active_config:
            return "Error: No active configuration. Run setup() first.", None

        config = self.active_config

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        logger.info(f"Generating with {config.model_name}...")
        logger.info(f"  Task: {task_type.value}")
        logger.info(f"  Device: {config.device.value}")
        logger.info(f"  Max tokens: {config.max_new_tokens}")

        start_time = time.time()

        # Generate text
        generated_text = self.model_manager.generate(
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            model_name=config.model_name
        )

        if not generated_text:
            return "Error: Generation failed", None

        generation_time = time.time() - start_time

        logger.info(f"Generation completed in {generation_time:.2f}s")

        # Validate if enabled and task is code generation
        validation_report = None

        if validate and task_type == TaskType.CODE_GENERATION and config.enable_validation:
            logger.info("Running validation pipeline...")

            # Prepare reviewer models
            reviewers = []
            if config.validator_model:
                reviewers.append((config.validator_model, config.device.value))

            # Validate
            validation_report = self.validator.validate(
                code=generated_text,
                mode=config.validation_mode,
                reviewer_models=reviewers if reviewers else None
            )

            logger.info(f"Validation: {validation_report.overall_pass and 'PASS' or 'FAIL'}")

        return generated_text, validation_report

    def interactive_generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.CODE_GENERATION
    ):
        """
        Interactive generation with validation feedback.

        Args:
            prompt: Input prompt
            task_type: Task type
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE GENERATION")
        print("=" * 70)

        if not self.active_config:
            print("Error: No active configuration. Run setup() first.")
            return

        print(f"\nModel: {self.active_config.model_name}")
        print(f"Device: {self.active_config.device.value}")
        print(f"Task: {task_type.value}")
        print(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # Generate
        print(f"\nGenerating...")
        generated_text, validation_report = self.generate(
            prompt,
            task_type=task_type
        )

        print(f"\n{'GENERATED OUTPUT':.<70}")
        print(generated_text)

        # Show validation if available
        if validation_report:
            print(f"\n{'VALIDATION RESULTS':.<70}")
            self.validator.print_report(validation_report)

        print("=" * 70 + "\n")

    def print_status(self):
        """Print engine status."""
        print("\n" + "=" * 70)
        print("INFERENCE ENGINE STATUS")
        print("=" * 70)

        # Hardware
        print(f"\n{'HARDWARE':.<50}")
        print(f"  Total Memory:        {self.hardware.get_total_memory_gb():>6.0f} GB")
        print(f"  Total TOPS:          {self.hardware.total_system_tops:>6.0f}")

        print(f"\n{'AVAILABLE DEVICES':.<50}")
        for device in self.list_available_devices():
            print(f"  • {device.value}")

        print(f"\n{'AVAILABLE MODELS':.<50}")
        for model in self.list_available_models():
            print(f"  • {model}")

        # Loaded models
        print(f"\n{'LOADED MODELS':.<50}")
        self.model_manager.print_status()

        # Active config
        if self.active_config:
            print(f"{'ACTIVE CONFIGURATION':.<50}")
            print(f"  Model:               {self.active_config.model_name}")
            print(f"  Device:              {self.active_config.device.value}")
            print(f"  Quantization:        {self.active_config.quantization.value}")
            print(f"  Max tokens:          {self.active_config.max_new_tokens}")
            print(f"  Validation:          {self.active_config.enable_validation and 'Enabled' or 'Disabled'}")
            if self.active_config.validator_model:
                print(f"  Validator:           {self.active_config.validator_model}")

        print("=" * 70 + "\n")


# Singleton instance
_engine: Optional[WhiteRabbitInferenceEngine] = None


def get_inference_engine() -> WhiteRabbitInferenceEngine:
    """Get or create singleton inference engine."""
    global _engine

    if _engine is None:
        _engine = WhiteRabbitInferenceEngine()

    return _engine


# Interactive CLI
def interactive_cli():
    """Interactive command-line interface."""
    engine = get_inference_engine()

    print("\n" + "=" * 70)
    print("WHITERABBITNEO INFERENCE ENGINE - INTERACTIVE CLI")
    print("=" * 70)
    print("\nCommands:")
    print("  setup <model>              - Setup with model")
    print("  device <device>            - Switch device")
    print("  generate <prompt>          - Generate text")
    print("  validate <code>            - Validate code")
    print("  status                     - Show status")
    print("  hardware                   - Show hardware")
    print("  quit                       - Exit")
    print("=" * 70 + "\n")

    while True:
        try:
            cmd = input("whiterabbit> ").strip()

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "quit" or command == "exit":
                print("Goodbye!")
                break

            elif command == "status":
                engine.print_status()

            elif command == "hardware":
                engine.hardware.print_summary()

            elif command == "setup":
                if not args:
                    print("Usage: setup <model_name>")
                    print(f"Available: {engine.list_available_models()}")
                    continue

                engine.setup(args)

            elif command == "device":
                if not args:
                    print("Usage: device <device_type>")
                    print(f"Available: {[d.value for d in engine.list_available_devices()]}")
                    continue

                try:
                    device = DeviceType(args)
                    engine.switch_device(device)
                except ValueError:
                    print(f"Invalid device: {args}")

            elif command == "generate":
                if not args:
                    print("Usage: generate <prompt>")
                    continue

                engine.interactive_generate(args)

            elif command == "help":
                print("\nCommands:")
                print("  setup <model>    - Setup with model")
                print("  device <device>  - Switch device")
                print("  generate <text>  - Generate text")
                print("  status           - Show status")
                print("  hardware         - Show hardware")
                print("  quit             - Exit\n")

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for commands")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_cli()
