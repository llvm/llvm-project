# LAT5150DRVMIL - Submodule Integration Guide

Complete guide for integrating LAT5150DRVMIL as a submodule in larger kernel/system build projects.

## Table of Contents

1. [Overview](#overview)
2. [Installation as Submodule](#installation-as-submodule)
3. [Python Package Integration](#python-package-integration)
4. [Entry Points and APIs](#entry-points-and-apis)
5. [Dependency Management](#dependency-management)
6. [Build System Integration](#build-system-integration)
7. [Configuration](#configuration)
8. [Testing Integration](#testing-integration)
9. [Examples](#examples)

---

## Overview

LAT5150DRVMIL is a complete AI platform that can be integrated as:
- **Git Submodule**: For source-level integration
- **Python Package**: Via setup.py/pip
- **System Component**: With entry points and services

### Architecture

```
Parent Project/
├── src/
├── modules/
│   └── LAT5150DRVMIL/          # ← Submodule
│       ├── __init__.py          # Main entry point
│       ├── ai_engine/           # AI engine module
│       ├── screenshot_intel/    # Screenshot intelligence module
│       ├── setup.py             # Python package setup
│       └── requirements.txt     # Dependencies
└── CMakeLists.txt / Makefile
```

---

## Installation as Submodule

### 1. Add as Git Submodule

```bash
# Add to your project
cd /path/to/your/project
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git modules/LAT5150DRVMIL

# Initialize and update
git submodule init
git submodule update

# Or combined:
git submodule update --init --recursive
```

### 2. Track Specific Version

```bash
cd modules/LAT5150DRVMIL
git checkout v1.0.0  # Or specific commit/tag
cd ../..
git add modules/LAT5150DRVMIL
git commit -m "Pin LAT5150DRVMIL to v1.0.0"
```

### 3. Update Submodule

```bash
git submodule update --remote modules/LAT5150DRVMIL
```

---

## Python Package Integration

### Install as Python Package

```bash
# From submodule directory
cd modules/LAT5150DRVMIL
pip install -e .

# Or with specific extras
pip install -e ".[screenshot_intel,telegram,api]"

# Or all extras
pip install -e ".[screenshot_intel,telegram,api,dev]"
```

### Install Dependencies Only

```bash
pip install -r modules/LAT5150DRVMIL/requirements.txt
```

### Use Without Installation

```python
import sys
sys.path.insert(0, 'modules/LAT5150DRVMIL')

from LAT5150DRVMIL import DSMILSystem

system = DSMILSystem()
```

---

## Entry Points and APIs

### Main System Interface

```python
from LAT5150DRVMIL import DSMILSystem

# Initialize complete system
system = DSMILSystem(
    enable_orchestrator=True,  # Multi-backend routing
    enable_screenshot_intel=True  # Screenshot analysis
)

# Generate AI response
response = system.generate("Explain quantum computing")

# Check system status
status = system.get_status()
print(status)
# {
#   'version': '1.0.0',
#   'components': ['ai_engine', 'orchestrator', 'screenshot_intel'],
#   'dsmil_available': True,
#   'orchestrator_available': True,
#   'screenshot_intel_available': True
# }
```

### AI Engine Only

```python
from LAT5150DRVMIL.ai_engine import DSMILAIEngine

# Local AI engine only (no cloud backends)
engine = DSMILAIEngine()

# Generate response
response = engine.generate(
    "Write a Python function to sort a list",
    model_selection="uncensored_code",  # or "creative", "balanced", etc.
    stream=False
)
```

### Unified Orchestrator

```python
from LAT5150DRVMIL.ai_engine import UnifiedAIOrchestrator

# Multi-backend routing (local + cloud fallback)
orchestrator = UnifiedAIOrchestrator()

# Automatic routing based on query
response = orchestrator.generate(
    "Analyze this image: [base64_data]",  # Auto-routes to Gemini
    model="auto"  # or "gemini", "openai", "local"
)
```

### Screenshot Intelligence

```python
from LAT5150DRVMIL.screenshot_intel import (
    ScreenshotIntelligence,
    VectorRAGSystem,
    SystemHealthMonitor
)

# Initialize screenshot intelligence
intel = ScreenshotIntelligence()

# Register device
intel.register_device(
    device_id="phone1",
    device_name="GrapheneOS Phone",
    device_type="grapheneos",
    screenshot_path="/path/to/screenshots"
)

# Ingest screenshots
result = intel.scan_device_screenshots("phone1")

# Search
results = intel.rag.search("error message", limit=10)

# Health monitoring
monitor = SystemHealthMonitor(screenshot_intel=intel)
health = monitor.run_health_check()
```

### System Validation

```python
from LAT5150DRVMIL.ai_engine import SystemValidator

# Validate entire system
validator = SystemValidator(detailed=True)
summary = validator.run_validation()

# summary contains:
# - total_checks
# - passed, warnings, failed, skipped
# - success_rate
```

---

## Dependency Management

### Minimal Dependencies (AI Engine Only)

```
pyyaml>=6.0
requests>=2.31.0
# + Ollama (installed separately)
```

### Full System Dependencies

See `requirements.txt` for complete list. Key categories:

**Core AI:**
- torch, transformers, accelerate
- intel-extension-for-pytorch, openvino

**Screenshot Intelligence:**
- qdrant-client, sentence-transformers
- paddleocr, paddlepaddle, pytesseract
- psutil, watchdog

**API/Services:**
- fastapi, uvicorn
- telethon (Telegram)

**Development:**
- pytest, black, mypy

### System Dependencies

```bash
# Docker (for Qdrant)
curl -fsSL https://get.docker.com | sh

# Tesseract OCR
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# System libraries
sudo apt-get install libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev

# Ollama (local AI)
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Build System Integration

### CMake Integration

```cmake
# CMakeLists.txt

# Add LAT5150DRVMIL submodule
add_subdirectory(modules/LAT5150DRVMIL EXCLUDE_FROM_ALL)

# Python components
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Add custom target for Python setup
add_custom_target(lat5150_python
    COMMAND ${Python3_EXECUTABLE} -m pip install -e modules/LAT5150DRVMIL
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Installing LAT5150DRVMIL Python package"
)

# Add to main build
add_dependencies(your_target lat5150_python)
```

### Makefile Integration

```makefile
# Makefile

.PHONY: all install-lat5150 test-lat5150

all: install-lat5150 your-build-targets

install-lat5150:
	cd modules/LAT5150DRVMIL && \
	pip install -r requirements.txt && \
	pip install -e .

test-lat5150:
	cd modules/LAT5150DRVMIL && \
	python3 02-ai-engine/system_validator.py

clean-lat5150:
	cd modules/LAT5150DRVMIL && \
	pip uninstall -y LAT5150DRVMIL
```

### Meson Integration

```meson
# meson.build

python = find_program('python3')

# Install LAT5150DRVMIL
run_command(
  python, '-m', 'pip', 'install', '-e',
  meson.source_root() / 'modules/LAT5150DRVMIL',
  check: true
)
```

---

## Configuration

### Environment Variables

```bash
# .env file or environment

# AI Engine
OLLAMA_HOST=http://localhost:11434
OPENAI_API_KEY=sk-...  # Optional
GEMINI_API_KEY=...  # Optional

# Screenshot Intelligence
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
SCREENSHOT_INTEL_API_KEY=...  # Optional

# Telegram Integration
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_PHONE=...

# Signal Integration
SIGNAL_PHONE=...
```

### Configuration File

```python
# config.py in your project

LAT5150_CONFIG = {
    'ai_engine': {
        'default_model': 'uncensored_code',
        'enable_cloud_fallback': False,
    },
    'screenshot_intel': {
        'data_dir': '/var/lib/lat5150/screenshots',
        'ocr_engine': 'paddleocr',  # or 'tesseract'
        'enable_health_monitoring': True,
    },
    'orchestrator': {
        'routing_strategy': 'auto',  # or 'local_only', 'cloud_prefer'
    }
}
```

---

## Testing Integration

### Validate System

```bash
# From your project root
python3 modules/LAT5150DRVMIL/02-ai-engine/system_validator.py

# Or via entry point (after pip install)
lat5150-validate --detailed
```

### Run Integration Tests

```bash
# Screenshot Intelligence tests
python3 modules/LAT5150DRVMIL/04-integrations/rag_system/test_screenshot_intel_integration.py -v

# Specific category
python3 modules/LAT5150DRVMIL/04-integrations/rag_system/test_screenshot_intel_integration.py \
  --test-category database -v
```

### In Your Test Suite

```python
import unittest
from LAT5150DRVMIL import DSMILSystem

class TestLAT5150Integration(unittest.TestCase):
    def setUp(self):
        self.system = DSMILSystem()

    def test_ai_generation(self):
        response = self.system.generate("Test prompt")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)

    def test_system_status(self):
        status = self.system.get_status()
        self.assertEqual(status['version'], '1.0.0')
        self.assertTrue(status['dsmil_available'])
```

---

## Examples

### Example 1: Simple AI Integration

```python
# simple_ai.py

from LAT5150DRVMIL import DSMILSystem

def main():
    # Initialize system
    system = DSMILSystem()

    # Generate response
    response = system.generate(
        "Explain the difference between TCP and UDP"
    )

    print(response)

if __name__ == "__main__":
    main()
```

### Example 2: Screenshot Analysis Pipeline

```python
# screenshot_pipeline.py

from LAT5150DRVMIL.screenshot_intel import (
    ScreenshotIntelligence,
    SystemHealthMonitor
)
from pathlib import Path

def main():
    # Initialize
    intel = ScreenshotIntelligence()
    monitor = SystemHealthMonitor(screenshot_intel=intel)

    # Register device
    intel.register_device(
        device_id="laptop1",
        device_name="Dell Latitude 5450",
        device_type="laptop",
        screenshot_path=Path("/home/user/screenshots")
    )

    # Ingest screenshots
    result = intel.scan_device_screenshots("laptop1")
    print(f"Ingested {result['screenshots_found']} screenshots")

    # Search for errors
    errors = intel.rag.search("error", limit=20, score_threshold=0.6)
    for result in errors:
        print(f"{result.score:.2f}: {result.document.filename}")

    # Health check
    health = monitor.run_health_check()
    print(f"System health: {health.overall_status}")

if __name__ == "__main__":
    main()
```

### Example 3: Multi-Component Integration

```python
# full_system.py

from LAT5150DRVMIL import DSMILSystem
from LAT5150DRVMIL.ai_engine import SystemValidator
from LAT5150DRVMIL.screenshot_intel import SystemHealthMonitor

def main():
    # Validate system first
    validator = SystemValidator(detailed=False)
    summary = validator.run_validation()

    if summary['failed'] > 0:
        print(f"⚠️ System validation failed: {summary['failed']} checks")
        return

    # Initialize full system
    system = DSMILSystem(
        enable_orchestrator=True,
        enable_screenshot_intel=True
    )

    # Check status
    status = system.get_status()
    print(f"System v{status['version']} initialized")
    print(f"Components: {', '.join(status['components'])}")

    # Use AI engine
    response = system.generate("List common network protocols")
    print(f"\nAI Response:\n{response}")

    # Health monitoring
    if system.screenshot_intel:
        monitor = SystemHealthMonitor(screenshot_intel=system.screenshot_intel)
        health = monitor.run_health_check()
        print(f"\nHealth: {health.overall_status}")

if __name__ == "__main__":
    main()
```

### Example 4: Build System Integration

```makefile
# Makefile

# Parent project Makefile

LAT5150_DIR = modules/LAT5150DRVMIL

.PHONY: all build test clean

all: init-lat5150 build

init-lat5150:
	@echo "Initializing LAT5150DRVMIL submodule..."
	git submodule update --init --recursive
	cd $(LAT5150_DIR) && pip install -r requirements.txt
	cd $(LAT5150_DIR) && pip install -e .

validate-lat5150:
	@echo "Validating LAT5150DRVMIL system..."
	python3 $(LAT5150_DIR)/02-ai-engine/system_validator.py

test: validate-lat5150
	@echo "Running LAT5150DRVMIL tests..."
	python3 $(LAT5150_DIR)/04-integrations/rag_system/test_screenshot_intel_integration.py

build: validate-lat5150
	@echo "Building parent project with LAT5150DRVMIL..."
	# Your build commands here

clean:
	@echo "Cleaning LAT5150DRVMIL..."
	cd $(LAT5150_DIR) && pip uninstall -y LAT5150DRVMIL
```

---

## API Reference

### Main Classes

#### `DSMILSystem`

Main system interface.

```python
class DSMILSystem:
    def __init__(
        self,
        enable_orchestrator: bool = False,
        enable_screenshot_intel: bool = False
    )

    def generate(self, prompt: str, **kwargs) -> str
    def get_status(self) -> dict
```

#### `DSMILAIEngine`

Core AI engine.

```python
class DSMILAIEngine:
    def generate(
        self,
        prompt: str,
        model_selection: str = "uncensored_code",
        stream: bool = False
    ) -> str
```

#### `ScreenshotIntelligence`

Screenshot analysis system.

```python
class ScreenshotIntelligence:
    def register_device(
        self,
        device_id: str,
        device_name: str,
        device_type: str,
        screenshot_path: Path
    )

    def ingest_screenshot(
        self,
        screenshot_path: Path,
        device_id: Optional[str] = None
    ) -> Dict

    def scan_device_screenshots(
        self,
        device_id: str,
        pattern: str = "*.png"
    ) -> Dict
```

#### `SystemValidator`

System validation tool.

```python
class SystemValidator:
    def __init__(self, detailed: bool = False)
    def run_validation(self) -> Dict[str, Any]
```

---

## Console Scripts

After `pip install`, these entry points are available:

```bash
# System validation
lat5150-validate
lat5150-validate --detailed
lat5150-validate --json

# Screenshot intelligence CLI
lat5150-screenshot-intel --help
lat5150-screenshot-intel device register phone1 "My Phone" grapheneos /path
lat5150-screenshot-intel search "error message"
lat5150-screenshot-intel timeline 2025-11-01 2025-11-12
```

---

## Troubleshooting

### Import Errors

```python
# If imports fail, ensure paths are correct:
import sys
from pathlib import Path

lat5150_path = Path(__file__).parent / "modules" / "LAT5150DRVMIL"
sys.path.insert(0, str(lat5150_path))
sys.path.insert(0, str(lat5150_path / "02-ai-engine"))
sys.path.insert(0, str(lat5150_path / "04-integrations" / "rag_system"))
```

### Missing Dependencies

```bash
# Reinstall all dependencies
cd modules/LAT5150DRVMIL
pip install -r requirements.txt

# Or with extras
pip install -e ".[screenshot_intel,api,dev]"
```

### Submodule Not Initialized

```bash
git submodule update --init --recursive
```

### System Validation Failures

```bash
# Check what's failing
python3 modules/LAT5150DRVMIL/02-ai-engine/system_validator.py --detailed

# Install missing dependencies based on recommendations
```

---

## Support

- **Documentation**: See `modules/LAT5150DRVMIL/06-intel-systems/`
- **API Reference**: This file
- **Production Guide**: `PRODUCTION_BEST_PRACTICES.md`
- **Integration Guide**: `INTEGRATION_GUIDE.md`

---

## Version

- LAT5150DRVMIL: v1.0.0
- Compatible with: Python 3.10+
- Platform: Linux (tested on Ubuntu 22.04/24.04)

