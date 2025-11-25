# Code Quality & Modularity

**LAT5150DRVMIL AI Engine**

**Version:** 1.0
**Last Updated:** 2025-11-21
**Owner:** Engineering Team

---

## Table of Contents

1. [Overview](#overview)
2. [CI/CD Pipeline](#cicd-pipeline)
3. [Dependency Management](#dependency-management)
4. [Code Organization](#code-organization)
5. [Code Quality Standards](#code-quality-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Development Workflow](#development-workflow)
8. [Tools & Automation](#tools--automation)

---

## 1. Overview

This document outlines the code quality standards, tools, and practices for the LAT5150DRVMIL AI Engine. Our approach emphasizes:

- **Automated Quality Checks**: CI/CD pipelines enforce standards
- **Dependency Stability**: Pinned versions for reproducible builds
- **Modular Architecture**: Utilities extracted into reusable components
- **Comprehensive Testing**: Unit, integration, and data validation tests
- **Clean Code**: Consistent formatting, type hints, and documentation

### Recent Improvements

**1. CI/CD Pipeline** (November 2025)
- Automated linting with flake8
- Code formatting with black
- Type checking with mypy
- Security scanning with bandit
- Complexity analysis with radon

**2. Dependency Management** (November 2025)
- Pinned all dependencies to specific versions
- Created `requirements-pinned.txt` for reproducible builds
- Separated dev dependencies in `requirements-dev.txt`

**3. Modular Architecture** (November 2025)
- Extracted `FileSearchUtility` from ResearchAgent
- Extracted `CodeAnalyzer` from ResearchAgent
- Improved testability via dependency injection

---

## 2. CI/CD Pipeline

### Pipeline Structure

We use GitHub Actions with three main workflows:

#### 2.1 Code Quality Workflow (`.github/workflows/code-quality.yml`)

**Triggered on**: Every push and pull request

**Checks performed**:

| Check | Tool | Purpose | Pass Criteria |
|-------|------|---------|---------------|
| **Linting (Errors)** | flake8 | Critical syntax/logic errors | Zero errors |
| **Linting (Warnings)** | flake8 | Code style violations | Max complexity 15, line length 120 |
| **Formatting** | black | Code formatting consistency | No changes needed |
| **Import Sorting** | isort | Import organization | Correct grouping & ordering |
| **Type Checking** | mypy | Type safety | No type errors |
| **Security Scanning** | bandit | Security vulnerabilities | No high/medium severity issues |
| **Complexity Analysis** | radon | Code complexity metrics | Cyclomatic complexity < 15 |

**Example flake8 configuration**:
```bash
# Critical errors (must pass)
flake8 02-ai-engine --count --select=E9,F63,F7,F82 --show-source --statistics

# Style warnings (should pass)
flake8 02-ai-engine --count --max-complexity=15 --max-line-length=120 --statistics
```

#### 2.2 Test Workflow (`.github/workflows/test.yml`)

**Triggered on**: Every push and pull request

**Test matrix**:
- **Operating Systems**: Ubuntu, macOS, Windows
- **Python Versions**: 3.9, 3.10, 3.11

**Test types**:
1. **Unit Tests**: Fast, isolated component tests
2. **Integration Tests**: Multi-component interaction tests
3. **Data Validation**: Sample data integrity checks

**Coverage requirements**:
- Minimum: 70% overall coverage
- Target: 85% overall coverage
- Critical modules: 90%+ coverage

#### 2.3 CI/CD Workflow (`.github/workflows/ci-cd.yml`)

**Full deployment pipeline**:

```
Quality Checks → Tests → Build → Deploy PyPI → Docker Build → Staging → Production
```

**Deployment triggers**:
- **Staging**: Pushes to `main` branch
- **Production**: Git tags matching `v*.*.*`

**Production safeguards**:
- Manual approval required
- All tests must pass
- Code quality checks must pass
- Security scan must pass

### Running Checks Locally

Before pushing code, run checks locally:

```bash
# Run all quality checks
make quality

# Or run individually
make lint          # Linting only
make format-check  # Check formatting
make typecheck     # Type checking
make security      # Security scan
```

---

## 3. Dependency Management

### Philosophy

We use **pinned dependencies** to ensure:
- **Reproducibility**: Same build every time
- **Stability**: No surprise breaking changes
- **Security**: Known, audited versions

### Dependency Files

| File | Purpose | Usage |
|------|---------|-------|
| `requirements.txt` | Core production dependencies | Unpinned (allows updates) |
| `requirements-pinned.txt` | **Pinned production dependencies** | **Use for deployments** |
| `requirements-dev.txt` | Development tools | Local development only |

### Pinned Dependencies (`requirements-pinned.txt`)

**Key dependencies** (as of 2025-11-21):

```txt
# Deep Learning & AI
torch==2.1.1
transformers==4.35.2
datasets==2.15.0
accelerate==0.25.0

# Data Management
dvc==3.35.2
pandas==2.1.3
numpy==1.26.2

# Testing & Quality
pytest==7.4.3
black==23.12.0
flake8==6.1.0
mypy==1.7.1
```

### Updating Dependencies

**Process**:

1. **Test updates locally**:
   ```bash
   # Create test environment
   python -m venv test-env
   source test-env/bin/activate

   # Install latest versions
   pip install -r requirements.txt

   # Run full test suite
   make test
   ```

2. **Pin working versions**:
   ```bash
   pip freeze > requirements-pinned.txt
   ```

3. **Update documentation**:
   - Update this file with new versions
   - Note any breaking changes
   - Update CHANGELOG.md

4. **Create PR**:
   - Title: "chore: Update dependencies (YYYY-MM-DD)"
   - Include test results
   - Note any behavioral changes

### Security Updates

**Critical vulnerabilities**:
- Update immediately (within 24 hours)
- Skip normal testing if necessary
- Create hotfix branch
- Deploy directly to production after smoke tests

**Non-critical vulnerabilities**:
- Follow normal update process
- Include in next regular update cycle

---

## 4. Code Organization

### Directory Structure

```
02-ai-engine/
├── ace_*.py                    # ACE-FCA core modules
├── utilities/                  # Shared utilities (NEW)
│   ├── __init__.py
│   ├── file_search.py         # File search operations
│   └── code_analyzer.py       # Code analysis utilities
├── performance/                # Performance optimization
│   ├── gpu_check.py
│   ├── performance_monitor.py
│   └── cache_manager.py
├── tests/                      # Test suite
│   ├── ace_test_utils.py      # Test utilities
│   ├── test_*.py              # Unit tests
│   └── integration/           # Integration tests
└── docs/                       # Documentation
```

### Module Responsibilities

#### ACE-FCA Core Modules

| Module | Responsibility | Dependencies |
|--------|----------------|--------------|
| `ace_config.py` | Configuration management | None |
| `ace_interfaces.py` | Abstract interfaces | None |
| `ace_exceptions.py` | Custom exceptions | None |
| `ace_registry.py` | Subagent registration | ace_exceptions |
| `ace_context_engine.py` | Context management | ace_config |
| `ace_phase_executor.py` | Phase execution | ace_interfaces, ace_config |
| `ace_subagents.py` | Specialized agents | utilities, ace_* |
| `ace_workflow_orchestrator.py` | Workflow orchestration | All ACE modules |

#### Utilities Package (NEW)

**Purpose**: Reusable, testable utility functions extracted from specialized agents.

**Modules**:

1. **`file_search.py`** - File discovery and search
   - **Extracted from**: ResearchAgent
   - **Responsibilities**:
     - Find files by pattern
     - Filter by content
     - Deduplicate results
   - **Key class**: `FileSearchUtility`

2. **`code_analyzer.py`** - Code structure analysis
   - **Extracted from**: ResearchAgent
   - **Responsibilities**:
     - Analyze architecture
     - Detect patterns (OOP, async, decorators)
     - Identify technology stack
   - **Key class**: `CodeAnalyzer`

**Benefits of extraction**:
- ✓ Single Responsibility Principle
- ✓ Easier to test in isolation
- ✓ Reusable across multiple agents
- ✓ ResearchAgent becomes a pure orchestrator

### Dependency Injection Pattern

**All classes use dependency injection** for flexibility and testability:

```python
# Good: Dependency injection with defaults
class ResearchAgent(BaseSubagent):
    def __init__(
        self,
        ai_engine,
        filesystem: Optional[FileSystemInterface] = None,
        command_executor: Optional[CommandExecutorInterface] = None,
        file_search: Optional[FileSearchUtility] = None,
        code_analyzer: Optional[CodeAnalyzer] = None
    ):
        # Use provided dependencies or create defaults
        self.filesystem = filesystem or StandardFileSystem()
        self.file_search = file_search or FileSearchUtility(
            filesystem=self.filesystem,
            command_executor=self.command_executor
        )
```

**Benefits**:
- Easy mocking in tests
- Runtime configuration flexibility
- Loose coupling between components

---

## 5. Code Quality Standards

### Python Code Style

We follow **PEP 8** with these modifications:

| Standard | Our Rule | Tool |
|----------|----------|------|
| Line length | 120 characters (not 79) | black, flake8 |
| String quotes | Double quotes preferred | black |
| Import sorting | stdlib → third-party → local | isort |
| Type hints | Required for public APIs | mypy |

### Formatting Standards

**Enforced by black**:

```python
# Good: Black-formatted code
def search_files(
    paths: List[str],
    patterns: List[str],
    query: Optional[str] = None,
    max_results: int = 50
) -> List[str]:
    """
    Search for files matching patterns.

    Args:
        paths: Directory paths to search
        patterns: File patterns (e.g., ["*.py"])
        query: Optional content filter
        max_results: Maximum files to return

    Returns:
        List of matching file paths
    """
    pass
```

**Format code**:
```bash
# Check formatting
make format-check

# Auto-format
make format
```

### Type Hints

**Required for**:
- All public functions/methods
- Function parameters and return types
- Class attributes

**Example**:
```python
from typing import List, Optional, Dict

class FileSearchUtility:
    def __init__(
        self,
        filesystem: Optional[FileSystemInterface] = None
    ):
        self.filesystem: FileSystemInterface = filesystem or StandardFileSystem()

    def search_files(
        self,
        paths: List[str],
        patterns: List[str]
    ) -> List[str]:
        ...
```

### Documentation Standards

**Docstrings required for**:
- All modules (file-level docstring)
- All classes
- All public functions/methods

**Format**: Google-style docstrings

```python
def analyze_architecture(files: List[str], max_dirs: int = 10) -> str:
    """
    Analyze architecture from file structure.

    Examines file paths to identify directory structure, API layers,
    test coverage, and common architectural patterns.

    Args:
        files: List of file paths to analyze
        max_dirs: Maximum directories to include in structure (default: 10)

    Returns:
        Human-readable architecture description

    Examples:
        >>> analyzer = CodeAnalyzer()
        >>> files = ["src/api/routes.py", "tests/test_api.py"]
        >>> arch = analyzer.analyze_architecture(files)
        >>> print(arch)
        - Structure: src/api, tests
        - API layer present
        - Test coverage exists
    """
    pass
```

### Complexity Limits

| Metric | Limit | Tool |
|--------|-------|------|
| Cyclomatic Complexity | 15 | flake8, radon |
| Function Length | 50 lines | Manual review |
| File Length | 1000 lines | Manual review |
| Parameters per Function | 7 | Manual review |

**High complexity code**:
- Refactor into smaller functions
- Use early returns
- Extract to separate classes

---

## 6. Testing Guidelines

### Test Organization

```
tests/
├── ace_test_utils.py          # Shared test utilities & mocks
├── test_ace_config.py          # Unit tests for ace_config
├── test_ace_subagents.py       # Unit tests for subagents
├── test_utilities.py           # Tests for utilities package
├── integration/
│   ├── test_workflow_e2e.py   # End-to-end workflow tests
│   └── test_performance.py    # Performance integration tests
└── data/
    └── test_data_validation.py # Data validation tests
```

### Test Categories

#### Unit Tests

**Characteristics**:
- Fast (< 100ms each)
- Isolated (no external dependencies)
- Deterministic (same input → same output)
- Use mocks for dependencies

**Example**:
```python
def test_file_search_with_query():
    """Test FileSearchUtility filters by query correctly"""
    # Setup mocks
    mock_fs = MockFileSystem({
        "file1.py": "def authenticate(): pass",
        "file2.py": "def process(): pass"
    })
    mock_cmd = MockCommandExecutor()

    # Create utility with mocks
    searcher = FileSearchUtility(
        filesystem=mock_fs,
        command_executor=mock_cmd
    )

    # Test
    results = searcher.search_files(["."], ["*.py"], query="authenticate")

    # Verify
    assert len(results) == 1
    assert "file1.py" in results
```

#### Integration Tests

**Characteristics**:
- Slower (< 5s each)
- Multi-component interactions
- May use real file system
- May use test databases

**Example**:
```python
def test_research_agent_with_real_files():
    """Test ResearchAgent with actual project files"""
    agent = create_subagent("research", ai_engine)

    result = agent.execute({
        "query": "authentication",
        "search_paths": ["02-ai-engine"],
        "file_patterns": ["*.py"]
    })

    assert result.success
    assert result.metadata["files_found"] > 0
```

#### Data Validation Tests

**Purpose**: Ensure sample data meets quality standards

```python
def test_sample_data_validation():
    """Validate all sample data files"""
    validator = DataValidator()

    for sample_file in glob.glob("sample_data/*.csv"):
        report = validator.validate_file(sample_file)
        assert report.is_valid, f"{sample_file} validation failed"
```

### Test Coverage

**Run coverage**:
```bash
# Generate coverage report
make coverage

# View HTML report
make coverage-html
open htmlcov/index.html
```

**Coverage targets**:
- Overall: 85%
- Critical modules (ace_workflow_orchestrator, ace_subagents): 90%
- Utilities (file_search, code_analyzer): 95%

---

## 7. Development Workflow

### Before Starting Work

1. **Update main branch**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements-pinned.txt
   pip install -r requirements-dev.txt
   ```

### During Development

1. **Write code with quality in mind**:
   - Follow code style guidelines
   - Add type hints
   - Write docstrings
   - Keep functions small

2. **Run checks frequently**:
   ```bash
   # Auto-format code
   make format

   # Run linter
   make lint

   # Run type checker
   make typecheck
   ```

3. **Write tests as you go**:
   - Unit test for each new function
   - Integration test for new features
   - Aim for 85%+ coverage

4. **Run tests frequently**:
   ```bash
   # Quick unit tests
   make test-unit

   # Full test suite
   make test
   ```

### Before Committing

**Run full quality check**:
```bash
make quality
```

This runs:
- Code formatting check
- Linting
- Type checking
- Security scanning
- Full test suite
- Coverage report

**All checks must pass** before committing.

### Creating a Pull Request

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Add file search utility"
   ```

2. **Push to remote**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

3. **Create PR on GitHub**:
   - Use descriptive title
   - Fill out PR template
   - Link related issues
   - Request reviewers

4. **Wait for CI checks**:
   - All workflows must pass
   - Fix any failures
   - Update PR as needed

5. **Address review comments**:
   - Make requested changes
   - Push updates
   - Re-request review

6. **Merge**:
   - Squash and merge (preferred)
   - Delete feature branch after merge

---

## 8. Tools & Automation

### Makefile Targets

Our `Makefile` provides 40+ automation targets:

#### Quality Checks

```bash
make quality         # Run all quality checks
make lint            # Run flake8 linter
make format          # Auto-format with black
make format-check    # Check formatting (no changes)
make typecheck       # Run mypy type checker
make security        # Run bandit security scanner
make complexity      # Analyze code complexity
```

#### Testing

```bash
make test            # Run all tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests
make coverage        # Generate coverage report
make coverage-html   # HTML coverage report
```

#### Data Management

```bash
make validate-data   # Validate all sample data
make dvc-status      # Check DVC status
make dvc-pull        # Pull data from remote
make dvc-push        # Push data to remote
```

#### Development

```bash
make install         # Install dependencies
make install-dev     # Install dev dependencies
make clean           # Clean generated files
make help            # Show all available targets
```

#### Performance

```bash
make profile         # Run profiler
make benchmark       # Run benchmarks
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to run checks automatically:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### IDE Integration

#### VS Code

**Recommended extensions**:
- Python (Microsoft)
- Pylance (Microsoft)
- Black Formatter (Microsoft)
- isort (Microsoft)
- Flake8 (Microsoft)

**Settings** (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true
}
```

#### PyCharm

**Settings**:
1. Preferences → Tools → Black → Enable
2. Preferences → Tools → External Tools → Add flake8
3. Preferences → Editor → Inspections → Enable mypy

---

## Best Practices Summary

### Do's ✓

- ✓ Use dependency injection with default values
- ✓ Write comprehensive docstrings
- ✓ Add type hints to all public APIs
- ✓ Keep functions small (< 50 lines)
- ✓ Write tests for new code
- ✓ Run `make quality` before committing
- ✓ Use meaningful variable names
- ✓ Extract reusable utilities
- ✓ Handle errors gracefully
- ✓ Log important operations

### Don'ts ✗

- ✗ Don't commit code that fails quality checks
- ✗ Don't push directly to main branch
- ✗ Don't skip writing tests
- ✗ Don't use generic exception handling
- ✗ Don't hardcode configuration values
- ✗ Don't mix concerns in single function
- ✗ Don't use mutable default arguments
- ✗ Don't ignore type errors
- ✗ Don't commit commented-out code
- ✗ Don't use `print()` for logging

---

## Troubleshooting

### Common Issues

**1. Flake8 fails with "line too long"**

```bash
# Solution: Auto-format with black
make format
```

**2. Black and flake8 disagree**

Our configuration is aligned. If this happens:
```bash
# Update tools
pip install --upgrade black flake8
```

**3. Mypy shows "missing type hints"**

Add type hints:
```python
# Before
def process(data):
    return data.upper()

# After
def process(data: str) -> str:
    return data.upper()
```

**4. Tests fail in CI but pass locally**

- Check Python version (use 3.10+ locally)
- Check dependencies match `requirements-pinned.txt`
- Look for environment-specific paths

**5. Import errors after refactoring**

```bash
# Ensure package is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/02-ai-engine"

# Or install in editable mode
pip install -e .
```

---

## Contact & Support

**Engineering Team**
Email: eng@lat5150drvmil.org
Issues: https://github.com/LAT5150DRVMIL/issues

**Code Quality Lead**
Slack: #code-quality
Wiki: https://wiki.lat5150drvmil.org/code-quality

---

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Black - The Uncompromising Code Formatter](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [mypy - Static Type Checker](https://mypy.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Document Version History**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-21 | Initial documentation | Engineering Team |
