# ACE-FCA Architectural Refactoring

## Overview

This document describes the architectural improvements made to the ACE-FCA (Advanced Context Engineering for Coding Agents) system to address coupling, testability, and maintainability issues.

## Problems Addressed

### From `ace_workflow_orchestrator.py`:
1. **Hardcoded Configuration/Prompts**: Phase prompts and model preferences were hardcoded in `__init__`
2. **Lack of Dependency Injection**: Direct instantiation of ACEContextEngine, no interface abstraction
3. **Multi-Responsibility**: Orchestrator handles workflow execution AND human review logic
4. **Test Code in Production Module**: MockAIEngine defined in __main__ section
5. **Limited Error Handling**: Generic exception handling without structured error information

### From `ace_subagents.py`:
1. **Direct External Command Execution**: ResearchAgent and VerifierAgent use `subprocess.run` directly
2. **Direct File I/O**: Multiple agents use `open()` and `os.path` directly
3. **Global Factory Dictionary**: Subagent factory uses global dictionary instead of registry pattern
4. **Test Utilities in Production Code**: MockAI class defined in __main__ section
5. **Tight Coupling**: Agents tightly coupled to shell commands and file system

## Solutions Implemented

### 1. Configuration Management (`ace_config.py`)

**Purpose**: Externalize all hardcoded configuration and prompts

**Key Components**:
- `PhaseType` enum for workflow phases
- `PhaseConfig` dataclass for phase-specific settings
- `ACEConfiguration` dataclass for complete system configuration
- JSON load/save support for flexible configuration management
- Default configurations for all workflow phases

**Benefits**:
- Configuration can be modified without code changes
- Easy to test with different configurations
- Supports multiple configuration profiles
- Clear separation of config from logic

**Usage**:
```python
from ace_config import load_configuration, save_configuration

# Load default or custom configuration
config = load_configuration("custom_config.json")

# Modify configuration
config.max_context_tokens = 16384
phase_config = config.get_phase_config(PhaseType.RESEARCH)
phase_config.max_tokens = 8000

# Save modified configuration
save_configuration(config, "modified_config.json")
```

### 2. Abstract Interfaces (`ace_interfaces.py`)

**Purpose**: Decouple subagents from direct dependencies on subprocess, file system, and human review

**Interfaces**:
- **FileSystemInterface**: Abstract file I/O operations
  - `read_file()`, `write_file()`, `file_exists()`, `list_files()`, `create_directory()`
  - Production: `StandardFileSystem`
  - Testing: `MockFileSystem`

- **CommandExecutorInterface**: Abstract command execution
  - `execute()`, `execute_async()`, `find_files()`
  - Production: `SubprocessCommandExecutor`
  - Testing: `MockCommandExecutor`

- **ReviewInterface**: Abstract human review mechanism
  - `request_review()`, `is_enabled()`, `set_auto_approve()`
  - Production: `InteractiveReview`
  - Testing: `MockReview`

**Benefits**:
- Complete isolation for unit testing (no real file I/O or subprocess calls)
- Easy to mock external dependencies
- Supports alternative implementations (e.g., remote file systems, container execution)
- Clear contracts for external operations

**Usage**:
```python
from ace_interfaces import create_production_interfaces, create_test_interfaces

# Production
fs, cmd, review = create_production_interfaces()

# Testing
mock_fs, mock_cmd, mock_review = create_test_interfaces(auto_approve=True)
mock_fs.write_file("/test/file.txt", "content")
mock_cmd.set_result("pytest", CommandResult(stdout="OK", stderr="", returncode=0, success=True))
```

### 3. Subagent Registry (`ace_registry.py`)

**Purpose**: Replace global factory dictionary with flexible, testable registry pattern

**Key Components**:
- `SubagentRegistry` class with capability-based discovery
- `SubagentMetadata` dataclass for agent information
- `SubagentCapability` enum (RESEARCH, PLANNING, IMPLEMENTATION, VERIFICATION, etc.)
- `@subagent` decorator for easy registration
- Factory override support for testing

**Benefits**:
- No global state (can create multiple registries)
- Supports dynamic registration/unregistration
- Capability-based agent discovery
- Priority-based selection
- Easy to test with mock agents

**Usage**:
```python
from ace_registry import SubagentRegistry, SubagentCapability, subagent

# Create registry
registry = SubagentRegistry()

# Register agent
registry.register(
    agent_type="custom",
    agent_class=CustomAgent,
    capabilities=[SubagentCapability.RESEARCH],
    priority=5
)

# Or use decorator
@subagent(
    agent_type="custom",
    capabilities=[SubagentCapability.RESEARCH],
    registry=registry
)
class CustomAgent(BaseSubagent):
    pass

# Create agent
agent = registry.create_agent("custom", ai_engine)

# Find by capability
research_agents = registry.find_by_capability(SubagentCapability.RESEARCH)
```

### 4. Phase Executor Abstraction (`ace_phase_executor.py`)

**Purpose**: Separate phase execution logic from orchestration (Single Responsibility Principle)

**Key Components**:
- `PhaseExecutor` abstract base class
- `SubagentPhaseExecutor` for production use
- `MockPhaseExecutor` for testing
- `PhaseExecutionPipeline` for managing phase sequences
- `PhaseInput`/`PhaseOutput` dataclasses for structured data

**Benefits**:
- Clear separation of concerns
- Each phase executor is independently testable
- Supports custom phase implementations
- Pipeline pattern for workflow management
- Standardized phase input/output format

**Usage**:
```python
from ace_phase_executor import (
    SubagentPhaseExecutor,
    PhaseExecutionPipeline,
    PhaseInput,
    create_standard_pipeline
)

# Create standard pipeline
pipeline = create_standard_pipeline(
    subagent_factory=create_subagent,
    compressor=compress_function
)

# Execute all phases
initial_input = PhaseInput(
    task_description="Add authentication",
    context={"project": "api"}
)

results = pipeline.execute_all(initial_input)

# Get summary
summary = pipeline.get_execution_summary()
```

### 5. Custom Exception Hierarchy (`ace_exceptions.py`)

**Purpose**: Provide structured error information with recovery hints

**Exception Hierarchy**:
```
ACEError (base)
├── ConfigurationError
│   ├── InvalidConfigurationError
│   └── MissingConfigurationError
├── ValidationError
│   ├── InvalidInputError
│   └── InvalidTaskError
├── ExecutionError
│   ├── PhaseExecutionError
│   ├── SubagentExecutionError
│   └── CommandExecutionError
├── ResourceError
│   ├── SubagentNotFoundError
│   ├── FileSystemError
│   └── ContextLimitError
├── IntegrationError
│   ├── AIEngineError
│   ├── APIError
│   └── MCPError
└── TimeoutError
    ├── PhaseTimeoutError
    └── CommandTimeoutError
```

**Features**:
- Structured error information (severity, category, context, recovery hints)
- `to_dict()` method for logging/serialization
- `wrap_exception()` utility for converting standard exceptions
- `format_error_for_logging()` for consistent logging format

**Benefits**:
- Clear error sources and categories
- Actionable recovery hints
- Better debugging with rich context
- Consistent error handling across system

**Usage**:
```python
from ace_exceptions import (
    SubagentNotFoundError,
    PhaseExecutionError,
    ErrorSeverity,
    format_error_for_logging
)

try:
    agent = create_subagent("unknown")
except Exception as e:
    raise SubagentNotFoundError(
        agent_type="unknown",
        available_types=["research", "planner"],
        severity=ErrorSeverity.HIGH
    )

try:
    result = phase.execute(task)
except Exception as e:
    raise PhaseExecutionError(
        phase_name="Research",
        reason="AI engine timeout",
        cause=e
    )
```

### 6. Test Utilities Module (`tests/ace_test_utils.py`)

**Purpose**: Centralize all test utilities and mocks

**Components**:
- `MockAI`: Simple mock AI for basic testing
- `MockAIEngine`: Realistic mock with phase-aware responses
- `MockSubagentResult`: Mock subagent results
- `MockWorkflowTask`: Mock workflow tasks
- Assertion helpers:
  - `assert_compressed_output()`: Verify compression limits
  - `assert_context_utilization()`: Verify 40-60% utilization
  - `assert_phase_output_format()`: Verify output format

**Benefits**:
- Clear separation of test code from production code
- Reusable test utilities
- Consistent mocking across tests
- Easier to maintain test infrastructure

**Usage**:
```python
from tests.ace_test_utils import (
    MockAIEngine,
    assert_compressed_output,
    assert_context_utilization
)

# Create mock AI with custom responses
ai = MockAIEngine(custom_responses={
    "authentication": "Custom auth response"
})

# Use in tests
orchestrator = ACEWorkflowOrchestrator(ai_engine=ai, enable_human_review=False)
result = orchestrator.execute_task(task)

# Verify compression
assert_compressed_output(result.compressed_output, max_tokens=500)

# Verify context utilization
assert_context_utilization(tokens_used=4000, max_tokens=8192)
```

## Refactored Files

### `ace_workflow_orchestrator.py` Changes

**Before**:
- Hardcoded prompts in `__init__`
- Direct review callback handling
- MockAIEngine defined in file
- Generic exception handling

**After**:
```python
def __init__(
    self,
    ai_engine,
    config: Optional[ACEConfiguration] = None,
    review_interface: Optional[ReviewInterface] = None,
    max_tokens: Optional[int] = None,
    enable_human_review: bool = True
):
    # Load configuration
    self.config = config or load_configuration()

    # Setup review interface
    self.review_interface = review_interface or InteractiveReview()

    # Phase prompts from configuration
    self.phase_prompts = {
        PhaseType.RESEARCH: self.config.get_phase_config(PhaseType.RESEARCH).prompt_template,
        # ...
    }
```

**Key Improvements**:
- Dependency injection for config and review interface
- Prompts loaded from configuration
- ReviewInterface abstraction for human review
- Structured error handling with ACEError
- MockAIEngine imported from tests module

### `ace_subagents.py` Changes

**Before**:
- Direct `subprocess.run()` calls
- Direct `open()` file I/O
- Global factory dictionary
- MockAI defined in file

**After**:
```python
class BaseSubagent(ABC):
    def __init__(
        self,
        ai_engine,
        max_tokens: int = 4096,
        filesystem: Optional[FileSystemInterface] = None,
        command_executor: Optional[CommandExecutorInterface] = None,
        review_interface: Optional[ReviewInterface] = None
    ):
        # Dependency injection with defaults
        self.filesystem = filesystem or StandardFileSystem()
        self.command_executor = command_executor or SubprocessCommandExecutor()
        self.review_interface = review_interface or InteractiveReview(auto_approve=True)
```

**ResearchAgent changes**:
```python
# Before
result = subprocess.run(["find", path, "-name", pattern, "-type", "f"], ...)
with open(file, 'r') as f:
    content = f.read()

# After
found_files = self.command_executor.find_files(directory=path, pattern=pattern, file_type="f")
content = self.filesystem.read_file(file)
```

**VerifierAgent changes**:
```python
# Before
result = subprocess.run(command.split(), capture_output=True, timeout=60)

# After
result = self.command_executor.execute(command, timeout=60)
```

**Registry integration**:
```python
# Register all subagents
register_subagent(
    agent_type="research",
    agent_class=ResearchAgent,
    capabilities=[SubagentCapability.RESEARCH, SubagentCapability.CODE_GENERATION],
    description="Codebase exploration and analysis",
    priority=10
)

# Create using registry
def create_subagent(agent_type: str, ai_engine, **kwargs) -> BaseSubagent:
    return registry_create_subagent(agent_type, ai_engine, **kwargs)
```

## Migration Guide

### For Existing Code

**Old way**:
```python
from ace_workflow_orchestrator import ACEWorkflowOrchestrator

orchestrator = ACEWorkflowOrchestrator(
    ai_engine=engine,
    max_tokens=8192,
    enable_human_review=True
)
```

**New way (backward compatible)**:
```python
from ace_workflow_orchestrator import ACEWorkflowOrchestrator

# Still works (uses defaults)
orchestrator = ACEWorkflowOrchestrator(
    ai_engine=engine,
    max_tokens=8192,
    enable_human_review=True
)

# Or with new features
from ace_config import load_configuration
from ace_interfaces import InteractiveReview

config = load_configuration("my_config.json")
review = InteractiveReview(auto_approve=False)

orchestrator = ACEWorkflowOrchestrator(
    ai_engine=engine,
    config=config,
    review_interface=review
)
```

### For Testing

**Old way**:
```python
class MockAIEngine:
    def generate(self, prompt, model="code", stream=False):
        return {"text": "Mock response"}

ai = MockAIEngine()
```

**New way**:
```python
from tests.ace_test_utils import MockAIEngine
from ace_interfaces import create_test_interfaces

# Realistic mock AI
ai = MockAIEngine()

# Mock interfaces for subagents
mock_fs, mock_cmd, mock_review = create_test_interfaces()

agent = ResearchAgent(
    ai_engine=ai,
    filesystem=mock_fs,
    command_executor=mock_cmd
)
```

## Testing Benefits

### Before
- Real subprocess calls during tests (slow, brittle)
- Real file I/O (requires cleanup, permissions)
- Hard to test error scenarios
- Inconsistent test utilities

### After
- No subprocess calls (fast, reliable)
- No real file I/O (clean, isolated)
- Easy to test error scenarios with mock results
- Centralized, reusable test utilities

**Example Test**:
```python
from tests.ace_test_utils import MockAIEngine
from ace_interfaces import create_test_interfaces

def test_research_agent_file_not_found():
    ai = MockAIEngine()
    mock_fs, mock_cmd, mock_review = create_test_interfaces()

    # Setup mock to return empty file list
    mock_cmd.set_result(
        "find . -name *.py -type f",
        CommandResult(stdout="", stderr="", returncode=0, success=True)
    )

    agent = ResearchAgent(
        ai_engine=ai,
        filesystem=mock_fs,
        command_executor=mock_cmd
    )

    result = agent.execute({
        "query": "test",
        "search_paths": ["."],
        "file_patterns": ["*.py"]
    })

    assert result.metadata['files_found'] == 0
    # No real subprocess calls were made!
```

## Performance Impact

- **Positive**: Tests run faster (no real I/O)
- **Neutral**: Production performance unchanged (same implementations)
- **Positive**: Easier to add performance optimizations (e.g., caching in interfaces)

## Future Enhancements

With these abstractions in place, we can easily add:

1. **Remote Execution**: Command executor that runs commands in containers/remote machines
2. **Distributed File Systems**: File system interface for S3, HDFS, etc.
3. **Async Review**: Non-blocking review interface with notifications
4. **Phase Caching**: Cache phase results to avoid re-execution
5. **Metrics Collection**: Track execution metrics through interfaces
6. **Alternative AI Engines**: Easy to swap AI engines with consistent interface

## Summary

This refactoring improves the ACE-FCA system by:

✅ **Decoupling**: Separating concerns using interfaces and abstractions
✅ **Testability**: Complete isolation with mock implementations
✅ **Maintainability**: Clear module boundaries and responsibilities
✅ **Flexibility**: Easy to extend with new implementations
✅ **Robustness**: Structured error handling with recovery hints
✅ **Configuration**: Externalized settings for easy customization

The changes are **backward compatible** - existing code continues to work while gaining access to new capabilities.
