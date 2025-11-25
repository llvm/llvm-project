#!/usr/bin/env python3
"""
ACE-FCA Configuration Module
-----------------------------
Externalized configuration for ACE-FCA workflow system including:
- Phase prompts
- Model preferences
- Context limits
- Compression settings
- Review thresholds

Addresses: Hardcoded configuration issue
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PhaseType(Enum):
    """Types of workflow phases"""
    RESEARCH = "research"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"


@dataclass
class PhaseConfig:
    """Configuration for a single phase"""
    name: str
    prompt_template: str
    model_preference: str
    max_tokens: int
    compression_ratio: float
    requires_review: bool
    timeout_seconds: Optional[int] = None


@dataclass
class ACEConfiguration:
    """Complete ACE-FCA system configuration"""
    # Context management
    max_context_tokens: int = 8192
    target_utilization_min: float = 0.4
    target_utilization_max: float = 0.6
    auto_compact: bool = True

    # Phase configurations
    phases: Dict[PhaseType, PhaseConfig] = None

    # Subagent settings
    subagent_max_tokens: int = 4096
    default_compression_ratio: float = 0.5

    # Review settings
    enable_human_review: bool = True
    review_on_compaction: bool = True
    review_timeout_seconds: int = 300

    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def __post_init__(self):
        if self.phases is None:
            self.phases = self._default_phases()

    def _default_phases(self) -> Dict[PhaseType, PhaseConfig]:
        """Create default phase configurations"""
        return {
            PhaseType.RESEARCH: PhaseConfig(
                name="Research",
                prompt_template="""
Analyze the codebase and gather information for the following task:

Task: {task}

Context: {context}

Focus on:
1. Relevant files and modules
2. Existing patterns and conventions
3. Dependencies and relationships
4. Potential challenges

Provide concise, actionable findings (max {max_tokens} tokens).
""",
                model_preference="quality_code",
                max_tokens=4096,
                compression_ratio=0.5,
                requires_review=False
            ),

            PhaseType.PLANNING: PhaseConfig(
                name="Planning",
                prompt_template="""
Create a detailed implementation plan for:

Task: {task}

Research Findings: {research_output}

Context: {context}

Create a plan that includes:
1. Step-by-step approach
2. Files to modify/create
3. Key functions/classes
4. Testing strategy
5. Potential risks

Provide structured plan (max {max_tokens} tokens).
""",
                model_preference="quality_code",
                max_tokens=4096,
                compression_ratio=0.5,
                requires_review=True
            ),

            PhaseType.IMPLEMENTATION: PhaseConfig(
                name="Implementation",
                prompt_template="""
Implement the following plan:

Task: {task}

Plan: {plan_output}

Context: {context}

Requirements:
1. Follow the plan exactly
2. Write clean, documented code
3. Maintain consistency with existing codebase
4. Include error handling

Provide implementation details (max {max_tokens} tokens).
""",
                model_preference="quality_code",
                max_tokens=6144,
                compression_ratio=0.5,
                requires_review=True
            ),

            PhaseType.VERIFICATION: PhaseConfig(
                name="Verification",
                prompt_template="""
Verify the implementation:

Task: {task}

Implementation: {implementation_output}

Context: {context}

Check:
1. Functionality correctness
2. Code quality and style
3. Test coverage
4. Performance considerations
5. Security implications

Provide verification report (max {max_tokens} tokens).
""",
                model_preference="quality_code",
                max_tokens=4096,
                compression_ratio=0.5,
                requires_review=False
            )
        }

    def get_phase_config(self, phase: PhaseType) -> PhaseConfig:
        """Get configuration for a specific phase"""
        return self.phases[phase]

    def update_phase_config(self, phase: PhaseType, **kwargs):
        """Update configuration for a specific phase"""
        config = self.phases[phase]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ACEConfiguration':
        """Create configuration from dictionary"""
        # Extract phase configs
        phases = {}
        if 'phases' in config_dict:
            for phase_name, phase_data in config_dict['phases'].items():
                phase_type = PhaseType(phase_name)
                phases[phase_type] = PhaseConfig(**phase_data)
            config_dict['phases'] = phases

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {
            'max_context_tokens': self.max_context_tokens,
            'target_utilization_min': self.target_utilization_min,
            'target_utilization_max': self.target_utilization_max,
            'auto_compact': self.auto_compact,
            'subagent_max_tokens': self.subagent_max_tokens,
            'default_compression_ratio': self.default_compression_ratio,
            'enable_human_review': self.enable_human_review,
            'review_on_compaction': self.review_on_compaction,
            'review_timeout_seconds': self.review_timeout_seconds,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'phases': {
                phase.value: {
                    'name': config.name,
                    'prompt_template': config.prompt_template,
                    'model_preference': config.model_preference,
                    'max_tokens': config.max_tokens,
                    'compression_ratio': config.compression_ratio,
                    'requires_review': config.requires_review,
                    'timeout_seconds': config.timeout_seconds
                }
                for phase, config in self.phases.items()
            }
        }
        return result


# Default global configuration
DEFAULT_CONFIG = ACEConfiguration()


def get_config() -> ACEConfiguration:
    """Get the global ACE configuration"""
    return DEFAULT_CONFIG


def set_config(config: ACEConfiguration):
    """Set the global ACE configuration"""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def load_config_from_file(filepath: str) -> ACEConfiguration:
    """Load configuration from JSON file"""
    import json
    from pathlib import Path

    config_path = Path(filepath)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return ACEConfiguration.from_dict(config_dict)


def save_config_to_file(config: ACEConfiguration, filepath: str):
    """Save configuration to JSON file"""
    import json
    from pathlib import Path

    config_path = Path(filepath)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Max context tokens: {config.max_context_tokens}")
    print(f"Number of phases: {len(config.phases)}")

    # Get specific phase config
    research_config = config.get_phase_config(PhaseType.RESEARCH)
    print(f"\nResearch phase model: {research_config.model_preference}")
    print(f"Research phase max tokens: {research_config.max_tokens}")
