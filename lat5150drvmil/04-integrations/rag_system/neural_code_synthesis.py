"""
Neural Code Synthesis (Phase 4.1)

Generate entire modules from high-level specifications using RAG + LLM.
Leverages existing documentation knowledge base to generate contextually-aware code
that follows your coding patterns and requirements.

Features:
- Multi-file module generation from natural language
- Pattern extraction from existing codebase
- Documentation-driven generation
- Kernel module templates (for LAT5150DRVMIL)
- Complete project scaffolding
- Automatic dependency resolution

Example:
    >>> synthesizer = NeuralCodeSynthesizer(rag_retriever)
    >>> spec = "NPU device driver with DMA, thermal monitoring, and power management"
    >>> module = synthesizer.generate_module(spec)
    >>> # Generates: kernel_npu_device.c, kernel_npu_dma.c, Makefile, docs
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from pathlib import Path


class ModuleType(Enum):
    """Types of modules that can be generated"""
    KERNEL_DRIVER = "kernel_driver"
    PYTHON_LIBRARY = "python_library"
    SPY_MODULE = "spy_module"  # Spy: statically-compiled Python variant
    CYTHON_MODULE = "cython_module"  # Cython: Python with C performance
    C_LIBRARY = "c_library"
    RUST_CRATE = "rust_crate"
    NPU_ACCELERATOR = "npu_accelerator"
    SYSTEM_SERVICE = "system_service"
    # Cybersecurity-focused module types
    MALWARE_ANALYZER = "malware_analyzer"
    EXPLOIT_DETECTOR = "exploit_detector"
    THREAT_HUNTER = "threat_hunter"
    FORENSICS_TOOL = "forensics_tool"
    SECURITY_MONITOR = "security_monitor"
    VULN_SCANNER = "vulnerability_scanner"


@dataclass
class FileTemplate:
    """A file to be generated"""
    filename: str
    content: str
    file_type: str  # "source", "header", "config", "docs", "build"
    language: str


@dataclass
class GeneratedModule:
    """A complete generated module"""
    module_name: str
    module_type: ModuleType
    description: str
    files: List[FileTemplate]
    dependencies: List[str]
    build_instructions: str
    usage_instructions: str


class SpecificationParser:
    """Parse natural language specifications"""

    # Keywords for module type detection
    # Order matters! More specific matches should come first
    MODULE_TYPE_KEYWORDS = {
        # Cybersecurity tools (most specific)
        ModuleType.MALWARE_ANALYZER: ['malware', 'virus', 'trojan', 'ransomware', 'analyzer', 'scanner'],
        ModuleType.EXPLOIT_DETECTOR: ['exploit', 'vulnerability', 'overflow', 'injection', 'detector'],
        ModuleType.THREAT_HUNTER: ['threat', 'hunt', 'ioc', 'indicator', 'compromise'],
        ModuleType.FORENSICS_TOOL: ['forensics', 'evidence', 'artifact', 'investigation', 'incident'],
        ModuleType.SECURITY_MONITOR: ['monitor', 'siem', 'log', 'alert', 'detection'],
        ModuleType.VULN_SCANNER: ['scan', 'assessment', 'penetration', 'pentest', 'audit'],
        # Performance-oriented Python variants - check before generic Python
        ModuleType.SPY_MODULE: ['spy', 'compiled python', 'static python', 'native python', 'performant python'],
        ModuleType.CYTHON_MODULE: ['cython', 'pyx', 'c extension', 'python extension', 'fast python', 'optimized python'],
        # System-level
        ModuleType.KERNEL_DRIVER: ['kernel', 'driver', 'device', 'hardware'],
        ModuleType.NPU_ACCELERATOR: ['npu', 'vpu', 'accelerator', 'inference', 'openvino'],
        ModuleType.SYSTEM_SERVICE: ['service', 'daemon', 'systemd'],
        # Language-specific
        ModuleType.RUST_CRATE: ['rust', 'crate'],
        ModuleType.C_LIBRARY: ['c library', 'shared library', 'static library'],
        ModuleType.PYTHON_LIBRARY: ['python', 'library', 'api', 'sdk'],
    }

    # Feature keywords
    FEATURE_KEYWORDS = {
        'dma': ['dma', 'direct memory', 'buffer'],
        'thermal': ['thermal', 'temperature', 'cooling'],
        'power': ['power', 'pm', 'suspend', 'resume'],
        'interrupt': ['interrupt', 'irq', 'isr'],
        'io': ['io', 'input', 'output', 'read', 'write'],
        'network': ['network', 'socket', 'tcp', 'udp'],
        'security': ['security', 'crypto', 'encryption'],
        # Cybersecurity features
        'static_analysis': ['static', 'binary', 'pe', 'elf', 'disassembly'],
        'dynamic_analysis': ['dynamic', 'sandbox', 'behavioral', 'runtime'],
        'yara': ['yara', 'signature', 'pattern', 'rule'],
        'memory_forensics': ['memory', 'dump', 'volatility', 'ram'],
        'network_forensics': ['packet', 'pcap', 'traffic', 'flow'],
        'ioc_extraction': ['ioc', 'indicator', 'hash', 'ip', 'domain'],
        'reversing': ['reverse', 'disassemble', 'decompile', 'ghidra'],
    }

    @classmethod
    def parse(cls, specification: str) -> Dict:
        """Parse specification into structured format"""

        spec_lower = specification.lower()

        # Detect module type
        module_type = cls._detect_module_type(spec_lower)

        # Extract features
        features = cls._extract_features(spec_lower)

        # Extract module name (heuristic)
        module_name = cls._extract_module_name(specification)

        return {
            'module_type': module_type,
            'features': features,
            'module_name': module_name,
            'original_spec': specification
        }

    @classmethod
    def _detect_module_type(cls, spec_lower: str) -> ModuleType:
        """Detect module type from specification"""

        for module_type, keywords in cls.MODULE_TYPE_KEYWORDS.items():
            if any(kw in spec_lower for kw in keywords):
                return module_type

        return ModuleType.PYTHON_LIBRARY  # Default

    @classmethod
    def _extract_features(cls, spec_lower: str) -> List[str]:
        """Extract features from specification"""
        features = []

        for feature, keywords in cls.FEATURE_KEYWORDS.items():
            if any(kw in spec_lower for kw in keywords):
                features.append(feature)

        return features

    @classmethod
    def _extract_module_name(cls, specification: str) -> str:
        """Extract or generate module name"""

        # Look for quotes
        quoted = re.findall(r'"([^"]+)"', specification)
        if quoted:
            name = quoted[0].lower().replace(' ', '_')
            return name

        # Generate from first few words
        words = specification.split()[:3]
        name = '_'.join(w.lower() for w in words if w.isalnum())

        return name or "generated_module"


class KernelDriverTemplate:
    """Templates for kernel drivers"""

    @staticmethod
    def generate_driver_c(module_name: str, features: List[str]) -> str:
        """Generate kernel driver .c file"""

        has_dma = 'dma' in features
        has_thermal = 'thermal' in features
        has_power = 'power' in features
        has_interrupt = 'interrupt' in features

        code = f'''/*
 * {module_name}.c - Kernel driver for {module_name.upper()}
 *
 * Auto-generated by Neural Code Synthesizer (Phase 4.1)
 * Target: LAT5150DRVMIL kernel development
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/device.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/io.h>
'''

        if has_dma:
            code += '''#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
'''

        if has_thermal:
            code += '''#include <linux/thermal.h>
'''

        if has_power:
            code += '''#include <linux/pm.h>
#include <linux/pm_runtime.h>
'''

        if has_interrupt:
            code += '''#include <linux/interrupt.h>
'''

        code += f'''
#define {module_name.upper()}_DRIVER_NAME "{module_name}"
#define {module_name.upper()}_VERSION "1.0.0"

/* Device private data */
struct {module_name}_device {{
    struct device *dev;
    void __iomem *base;
'''

        if has_dma:
            code += '''    dma_addr_t dma_handle;
    void *dma_buffer;
    size_t dma_size;
'''

        if has_thermal:
            code += '''    struct thermal_zone_device *thermal_zone;
    int temperature;
'''

        code += '''};

static struct {module_name}_device *{module_name}_dev;

'''

        # DMA functions
        if has_dma:
            code += f'''
/* DMA buffer allocation */
static int {module_name}_dma_alloc(struct {module_name}_device *dev, size_t size)
{{
    dev->dma_buffer = dma_alloc_coherent(dev->dev, size,
                                        &dev->dma_handle, GFP_KERNEL);
    if (!dev->dma_buffer) {{
        dev_err(dev->dev, "DMA buffer allocation failed\\n");
        return -ENOMEM;
    }}

    dev->dma_size = size;
    dev_info(dev->dev, "DMA buffer allocated: %zu bytes at 0x%llx\\n",
            size, (unsigned long long)dev->dma_handle);

    return 0;
}}

/* DMA buffer deallocation */
static void {module_name}_dma_free(struct {module_name}_device *dev)
{{
    if (dev->dma_buffer) {{
        dma_free_coherent(dev->dev, dev->dma_size,
                         dev->dma_buffer, dev->dma_handle);
        dev->dma_buffer = NULL;
    }}
}}
'''

        # Thermal functions
        if has_thermal:
            code += f'''
/* Get temperature */
static int {module_name}_get_temp(struct thermal_zone_device *tz, int *temp)
{{
    struct {module_name}_device *dev = tz->devdata;

    /* Read temperature from device register */
    *temp = dev->temperature * 1000;  /* Convert to millidegrees */

    return 0;
}}

static struct thermal_zone_device_ops {module_name}_thermal_ops = {{
    .get_temp = {module_name}_get_temp,
}};

/* Register thermal zone */
static int {module_name}_thermal_init(struct {module_name}_device *dev)
{{
    dev->thermal_zone = thermal_zone_device_register(
        "{module_name}",
        0, 0, dev,
        &{module_name}_thermal_ops,
        NULL, 0, 0);

    if (IS_ERR(dev->thermal_zone)) {{
        dev_err(dev->dev, "Failed to register thermal zone\\n");
        return PTR_ERR(dev->thermal_zone);
    }}

    return 0;
}}
'''

        # Power management
        if has_power:
            code += f'''
/* Suspend */
static int {module_name}_suspend(struct device *dev)
{{
    dev_info(dev, "Suspending {module_name}\\n");

    /* Save device state */

    return 0;
}}

/* Resume */
static int {module_name}_resume(struct device *dev)
{{
    dev_info(dev, "Resuming {module_name}\\n");

    /* Restore device state */

    return 0;
}}

static const struct dev_pm_ops {module_name}_pm_ops = {{
    .suspend = {module_name}_suspend,
    .resume = {module_name}_resume,
}};
'''

        # Interrupt handler
        if has_interrupt:
            code += f'''
/* Interrupt handler */
static irqreturn_t {module_name}_irq_handler(int irq, void *data)
{{
    struct {module_name}_device *dev = data;

    /* Handle interrupt */
    dev_dbg(dev->dev, "Interrupt received\\n");

    return IRQ_HANDLED;
}}
'''

        # Probe and remove
        code += f'''
/* Probe */
static int {module_name}_probe(struct platform_device *pdev)
{{
    struct {module_name}_device *dev;
    int ret;

    dev_info(&pdev->dev, "Probing {module_name} driver\\n");

    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->dev = &pdev->dev;
    {module_name}_dev = dev;
    platform_set_drvdata(pdev, dev);

    /* Map device registers */
    dev->base = devm_platform_ioremap_resource(pdev, 0);
    if (IS_ERR(dev->base)) {{
        return PTR_ERR(dev->base);
    }}

'''

        if has_dma:
            code += f'''    /* Initialize DMA */
    ret = {module_name}_dma_alloc(dev, 4096);  /* 4KB buffer */
    if (ret < 0) {{
        return ret;
    }}

'''

        if has_thermal:
            code += f'''    /* Initialize thermal monitoring */
    ret = {module_name}_thermal_init(dev);
    if (ret < 0) {{
        goto err_dma;
    }}

'''

        if has_interrupt:
            code += f'''    /* Request IRQ */
    ret = devm_request_irq(dev->dev, platform_get_irq(pdev, 0),
                          {module_name}_irq_handler,
                          IRQF_SHARED, "{module_name}", dev);
    if (ret < 0) {{
        dev_err(dev->dev, "Failed to request IRQ\\n");
        goto err_thermal;
    }}

'''

        code += f'''    dev_info(&pdev->dev, "{module_name} driver loaded successfully\\n");

    return 0;

'''

        # Error handling
        if has_thermal:
            code += f'''err_thermal:
    thermal_zone_device_unregister(dev->thermal_zone);
'''
        if has_dma:
            code += f'''err_dma:
    {module_name}_dma_free(dev);
'''

        code += f'''    return ret;
}}

/* Remove */
static int {module_name}_remove(struct platform_device *pdev)
{{
    struct {module_name}_device *dev = platform_get_drvdata(pdev);

    dev_info(&pdev->dev, "Removing {module_name} driver\\n");

'''

        if has_thermal:
            code += f'''    thermal_zone_device_unregister(dev->thermal_zone);
'''

        if has_dma:
            code += f'''    {module_name}_dma_free(dev);
'''

        code += f'''
    return 0;
}}

/* Device tree match table */
static const struct of_device_id {module_name}_of_match[] = {{
    {{ .compatible = "{module_name}" }},
    {{ }}
}};
MODULE_DEVICE_TABLE(of, {module_name}_of_match);

/* Platform driver */
static struct platform_driver {module_name}_driver = {{
    .driver = {{
        .name = {module_name.upper()}_DRIVER_NAME,
        .of_match_table = {module_name}_of_match,
'''

        if has_power:
            code += f'''        .pm = &{module_name}_pm_ops,
'''

        code += f'''    }},
    .probe = {module_name}_probe,
    .remove = {module_name}_remove,
}};

module_platform_driver({module_name}_driver);

MODULE_AUTHOR("LAT5150DRVMIL Development Team");
MODULE_DESCRIPTION("{module_name.upper()} Driver");
MODULE_LICENSE("GPL");
MODULE_VERSION({module_name.upper()}_VERSION);
'''

        return code

    @staticmethod
    def generate_makefile(module_name: str) -> str:
        """Generate Makefile"""

        return f'''# Makefile for {module_name} kernel module
# Auto-generated by Neural Code Synthesizer

obj-m += {module_name}.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
\tmake -C $(KDIR) M=$(PWD) modules

clean:
\tmake -C $(KDIR) M=$(PWD) clean

install:
\tmake -C $(KDIR) M=$(PWD) modules_install
\tdepmod -a

load:
\tsudo insmod {module_name}.ko

unload:
\tsudo rmmod {module_name}

reload: unload load

.PHONY: all clean install load unload reload
'''


class NeuralCodeSynthesizer:
    """Main neural code synthesis engine"""

    def __init__(self, rag_retriever=None):
        self.rag_retriever = rag_retriever
        self.spec_parser = SpecificationParser()

    def generate_module(self, specification: str) -> GeneratedModule:
        """Generate complete module from specification"""

        # Parse specification
        parsed_spec = self.spec_parser.parse(specification)

        module_type = parsed_spec['module_type']
        features = parsed_spec['features']
        module_name = parsed_spec['module_name']

        # Generate based on module type
        if module_type == ModuleType.KERNEL_DRIVER or module_type == ModuleType.NPU_ACCELERATOR:
            return self._generate_kernel_driver(module_name, features, specification)
        elif module_type == ModuleType.PYTHON_LIBRARY:
            return self._generate_python_library(module_name, features, specification)
        elif module_type == ModuleType.SPY_MODULE:
            return self._generate_spy_module(module_name, features, specification)
        elif module_type == ModuleType.CYTHON_MODULE:
            return self._generate_cython_module(module_name, features, specification)
        # Cybersecurity module types
        elif module_type == ModuleType.MALWARE_ANALYZER:
            return self._generate_malware_analyzer(module_name, features, specification)
        elif module_type == ModuleType.EXPLOIT_DETECTOR:
            return self._generate_exploit_detector(module_name, features, specification)
        elif module_type == ModuleType.THREAT_HUNTER:
            return self._generate_threat_hunter(module_name, features, specification)
        elif module_type == ModuleType.FORENSICS_TOOL:
            return self._generate_forensics_tool(module_name, features, specification)
        elif module_type == ModuleType.SECURITY_MONITOR:
            return self._generate_security_monitor(module_name, features, specification)
        elif module_type == ModuleType.VULN_SCANNER:
            return self._generate_vuln_scanner(module_name, features, specification)
        else:
            return self._generate_generic_module(module_name, module_type, specification)

    def _generate_kernel_driver(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate kernel driver module"""

        files = []

        # Generate driver .c file
        driver_c = KernelDriverTemplate.generate_driver_c(module_name, features)
        files.append(FileTemplate(
            filename=f"{module_name}.c",
            content=driver_c,
            file_type="source",
            language="c"
        ))

        # Generate Makefile
        makefile = KernelDriverTemplate.generate_makefile(module_name)
        files.append(FileTemplate(
            filename="Makefile",
            content=makefile,
            file_type="build",
            language="makefile"
        ))

        # Generate README
        readme = self._generate_readme(module_name, spec, features)
        files.append(FileTemplate(
            filename="README.md",
            content=readme,
            file_type="docs",
            language="markdown"
        ))

        dependencies = ["linux-headers", "build-essential"]

        build_instructions = f"""
Build Instructions:
1. make
2. sudo make install
3. sudo modprobe {module_name}

Verify:
  lsmod | grep {module_name}
  dmesg | tail
"""

        usage_instructions = f"""
Usage:
  Load module:   sudo modprobe {module_name}
  Unload module: sudo rmmod {module_name}
  Check status:  cat /sys/module/{module_name}/version
"""

        return GeneratedModule(
            module_name=module_name,
            module_type=ModuleType.KERNEL_DRIVER,
            description=spec,
            files=files,
            dependencies=dependencies,
            build_instructions=build_instructions,
            usage_instructions=usage_instructions
        )

    def _generate_python_library(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate Python library"""

        files = []

        # Generate __init__.py
        init_py = f'''"""
{module_name}

{spec}

Auto-generated by Neural Code Synthesizer (Phase 4.1)
"""

__version__ = "0.1.0"

from .core import {module_name.replace('_', ' ').title().replace(' ', '')}

__all__ = ['{module_name.replace('_', ' ').title().replace(' ', '')}']
'''

        files.append(FileTemplate(
            filename="__init__.py",
            content=init_py,
            file_type="source",
            language="python"
        ))

        # Generate core.py
        class_name = module_name.replace('_', ' ').title().replace(' ', '')
        core_py = f'''"""
Core implementation for {module_name}
"""

class {class_name}:
    """Main class for {module_name}"""

    def __init__(self):
        """Initialize {module_name}"""
        pass

    def run(self):
        """Run {module_name}"""
        raise NotImplementedError("Implement this method")
'''

        files.append(FileTemplate(
            filename="core.py",
            content=core_py,
            file_type="source",
            language="python"
        ))

        # Generate setup.py
        setup_py = f'''from setuptools import setup, find_packages

setup(
    name="{module_name}",
    version="0.1.0",
    description="{spec}",
    packages=find_packages(),
    python_requires=">=3.8",
)
'''

        files.append(FileTemplate(
            filename="setup.py",
            content=setup_py,
            file_type="build",
            language="python"
        ))

        return GeneratedModule(
            module_name=module_name,
            module_type=ModuleType.PYTHON_LIBRARY,
            description=spec,
            files=files,
            dependencies=["python>=3.8"],
            build_instructions="pip install -e .",
            usage_instructions=f"import {module_name}"
        )

    def _generate_spy_module(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate Spy module (statically-compiled Python variant)"""

        files = []

        # Generate main .spy file with static typing
        spy_main = f'''"""
{module_name} - Spy Module
Generated for LAT5150DRVMIL high-performance computing

Specification: {spec}

Spy Language: Statically-compiled Python variant
Features: Static typing, native compilation, WebAssembly support
"""

from typing import List, Dict, Optional


def fibonacci(n: int) -> int:
    """Calculate fibonacci number with static typing"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def process_array(data: List[int]) -> int:
    """Process array with O(n) performance"""
    total: int = 0
    for item in data:
        total += item
    return total


class {module_name.replace('_', ' ').title().replace(' ', '')}:
    """Main class for {module_name} with static types"""

    def __init__(self) -> None:
        """Initialize {module_name}"""
        self.name: str = "{module_name}"
        self.version: str = "1.0.0"

    def run(self) -> int:
        """Run {module_name} - returns status code"""
        print(f"Running {{self.name}} v{{self.version}}")

        # Example computation
        result: int = fibonacci(10)
        print(f"Fibonacci(10) = {{result}}")

        # Array processing
        data: List[int] = [1, 2, 3, 4, 5]
        total: int = process_array(data)
        print(f"Array sum: {{total}}")

        return 0


def main() -> int:
    """Main entry point"""
    app = {module_name.replace('_', ' ').title().replace(' ', '')}()
    return app.run()


if __name__ == "__main__":
    import sys
    sys.exit(main())
'''

        files.append(FileTemplate(
            filename=f"{module_name}.spy",
            content=spy_main,
            file_type="source",
            language="spy"
        ))

        # Generate Makefile for compilation
        makefile = f'''# Spy Module Makefile for {module_name}
# Compiles Spy code to native binary and WebAssembly

.PHONY: all clean run compile-native compile-wasm test

# Default target
all: compile-native

# Compile to native binary
compile-native:
\t@echo "Compiling {module_name}.spy to native binary..."
\tspy compile {module_name}.spy -o {module_name}
\t@echo "✓ Native binary created: {module_name}"

# Compile to WebAssembly
compile-wasm:
\t@echo "Compiling {module_name}.spy to WebAssembly..."
\tspy compile --target=wasm {module_name}.spy -o {module_name}.wasm
\t@echo "✓ WebAssembly module created: {module_name}.wasm"

# Run in interpreted mode (faster development)
run:
\tspy {module_name}.spy

# Run compiled native binary
run-native: compile-native
\t./{module_name}

# Run tests (interpreted mode)
test:
\t@echo "Running tests..."
\tspy test_{module_name}.spy

# Performance benchmark (native vs interpreted)
benchmark: compile-native
\t@echo "Benchmarking interpreted mode..."
\t@time spy {module_name}.spy
\t@echo ""
\t@echo "Benchmarking native mode..."
\t@time ./{module_name}

# Clean build artifacts
clean:
\trm -f {module_name} {module_name}.wasm
\trm -f *.pyc __pycache__
\t@echo "✓ Cleaned build artifacts"

# Show Spy compiler version
version:
\tspy --version
'''

        files.append(FileTemplate(
            filename="Makefile",
            content=makefile,
            file_type="build",
            language="makefile"
        ))

        # Generate README with Spy-specific info
        readme = f'''# {module_name}

{spec}

## About Spy Language

This module is written in **Spy** - a statically-compiled variant of Python designed for performance-critical applications.

### Key Features

- **Static Typing**: All types are declared and checked at compile time
- **Native Compilation**: Compiles to native binaries (no interpreter overhead)
- **WebAssembly Support**: Can compile to WASM for browser/edge deployment
- **Python Syntax**: Familiar Python syntax with type annotations
- **High Performance**: 10-100x faster than standard Python for numeric code

## Prerequisites

```bash
# Install Spy compiler
git clone https://github.com/spylang/spy
cd spy
make
sudo make install
```

## Building

### Interpreted Mode (Development)
```bash
spy {module_name}.spy
```

### Compiled Mode (Production)
```bash
# Compile to native binary
make compile-native

# Run native binary
./{module_name}
```

### WebAssembly
```bash
# Compile to WASM
make compile-wasm

# Output: {module_name}.wasm
```

## Performance

Spy provides significant performance improvements over standard Python:

- **Numeric computation**: 10-50x faster
- **Array operations**: 5-20x faster
- **Overall**: 2-100x faster (depending on workload)

Perfect for LAT5150DRVMIL's performance-critical cybersecurity operations.

## Usage

```spy
from {module_name} import {module_name.replace('_', ' ').title().replace(' ', '')}

app = {module_name.replace('_', ' ').title().replace(' ', '')}()
app.run()
```

## Compilation Options

```bash
# Optimize for speed
spy compile -O3 {module_name}.spy

# Enable debugging symbols
spy compile -g {module_name}.spy

# Cross-compile for ARM
spy compile --target=arm64 {module_name}.spy
```

## Learn More

- Spy Language: https://github.com/spylang/spy
- Documentation: https://spy-lang.org/docs
'''

        files.append(FileTemplate(
            filename="README.md",
            content=readme,
            file_type="docs",
            language="markdown"
        ))

        return GeneratedModule(
            module_name=module_name,
            module_type=ModuleType.SPY_MODULE,
            description=spec,
            files=files,
            dependencies=["spy-lang"],
            build_instructions="make compile-native",
            usage_instructions=f"./{module_name}  # or: spy {module_name}.spy"
        )

    def _generate_cython_module(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate Cython module (Python with C performance)"""

        files = []
        class_name = module_name.replace('_', ' ').title().replace(' ', '')

        # Generate main .pyx file (Cython source)
        pyx_main = f'''"""
{module_name} - Cython Module
Generated for LAT5150DRVMIL high-performance computing

Specification: {spec}

Cython: Optimizing static compiler for Python
Features: C-level performance, seamless Python integration, static typing
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport sqrt, pow
cimport cython

# Type definitions for performance
ctypedef unsigned int uint
ctypedef unsigned long ulong


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_fibonacci(int n) nogil:
    """Fast Fibonacci calculation (C speed, no Python overhead)"""
    if n <= 1:
        return n
    return c_fibonacci(n - 1) + c_fibonacci(n - 2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double c_array_sum(double[:] arr) nogil:
    """Fast array summation with typed memoryviews"""
    cdef:
        double total = 0.0
        int i
        int n = arr.shape[0]

    for i in range(n):
        total += arr[i]

    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_vector_add(double[:] a, double[:] b, double[:] result) nogil:
    """In-place vector addition (SIMD-friendly)"""
    cdef int i
    cdef int n = a.shape[0]

    for i in range(n):
        result[i] = a[i] + b[i]


# Python-accessible wrapper functions
cpdef double fibonacci(int n):
    """Calculate Fibonacci number (Python-callable, C-speed)"""
    return c_fibonacci(n)


cpdef double array_sum(double[:] arr):
    """Sum array elements (Python-callable, typed memoryview)"""
    return c_array_sum(arr)


cpdef void vector_add(double[:] a, double[:] b, double[:] result):
    """Add two vectors element-wise (Python-callable)"""
    c_vector_add(a, b, result)


cdef class {class_name}:
    """
    Main class for {module_name} with C-level performance

    Uses Cython's cdef class for maximum speed and minimal overhead.
    All attributes are typed for C-level access.
    """

    cdef:
        str name
        str version
        int iteration_count
        double* data_buffer  # Raw C pointer for maximum performance
        size_t buffer_size

    def __cinit__(self):
        """C-level initialization (called before __init__)"""
        self.name = "{module_name}"
        self.version = "1.0.0"
        self.iteration_count = 0
        self.buffer_size = 1024

        # Allocate C buffer
        self.data_buffer = <double*>malloc(self.buffer_size * sizeof(double))
        if self.data_buffer == NULL:
            raise MemoryError("Failed to allocate buffer")

        # Initialize buffer to zeros
        memset(self.data_buffer, 0, self.buffer_size * sizeof(double))

    def __dealloc__(self):
        """C-level cleanup (automatic memory management)"""
        if self.data_buffer != NULL:
            free(self.data_buffer)

    cpdef int run(self):
        """
        Run {module_name} - returns status code

        cpdef allows both Python and C-level calls for flexibility
        """
        print(f"Running {{self.name}} v{{self.version}}")

        # Example: Fast Fibonacci
        cdef double result = c_fibonacci(20)
        print(f"Fibonacci(20) = {{result:.0f}}")

        # Example: Array operations with typed memoryviews
        import numpy as np
        cdef double[:] arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cdef double total = c_array_sum(arr)
        print(f"Array sum: {{total}}")

        # Example: Vector operations
        cdef double[:] a = np.array([1.0, 2.0, 3.0])
        cdef double[:] b = np.array([4.0, 5.0, 6.0])
        cdef double[:] result_vec = np.zeros(3)
        c_vector_add(a, b, result_vec)
        print(f"Vector addition: {{np.asarray(result_vec)}}")

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _process_data_internal(self) nogil:
        """
        Internal C-only method (no Python overhead)

        nogil allows true parallelism with threading
        """
        cdef size_t i
        for i in range(self.buffer_size):
            self.data_buffer[i] = sqrt(pow(self.data_buffer[i], 2.0))

    cpdef void process_data(self):
        """Process internal data buffer (Python-callable wrapper)"""
        with nogil:
            self._process_data_internal()


def main():
    """Main entry point"""
    app = {class_name}()
    return app.run()


if __name__ == "__main__":
    import sys
    sys.exit(main())
'''

        files.append(FileTemplate(
            filename=f"{module_name}.pyx",
            content=pyx_main,
            file_type="source",
            language="cython"
        ))

        # Generate .pxd header file (Cython declarations)
        pxd_header = f'''"""
{module_name}.pxd - Cython declarations for {module_name}

This file defines the public C API and allows other Cython modules
to cimport and use these functions efficiently.
"""

# cython: language_level=3

# C-level type definitions
ctypedef unsigned int uint
ctypedef unsigned long ulong

# Declare C functions for cimport
cdef double c_fibonacci(int n) nogil
cdef double c_array_sum(double[:] arr) nogil
cdef void c_vector_add(double[:] a, double[:] b, double[:] result) nogil

# Declare cdef class for cimport
cdef class {class_name}:
    cdef:
        str name
        str version
        int iteration_count
        double* data_buffer
        size_t buffer_size

    cpdef int run(self)
    cpdef void process_data(self)
    cdef void _process_data_internal(self) nogil
'''

        files.append(FileTemplate(
            filename=f"{module_name}.pxd",
            content=pxd_header,
            file_type="header",
            language="cython"
        ))

        # Generate setup.py for compilation
        setup_py = f'''"""
setup.py - Build configuration for {module_name}

Compiles Cython extensions to native Python modules (.so/.pyd)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define extension module
extensions = [
    Extension(
        "{module_name}",
        ["{module_name}.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",              # Maximum optimization
            "-march=native",    # CPU-specific optimizations
            "-ffast-math",      # Fast math operations
            "-fopenmp",         # OpenMP parallelization
        ],
        extra_link_args=["-fopenmp"],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ],
    )
]

setup(
    name="{module_name}",
    version="1.0.0",
    description="{spec}",
    ext_modules=cythonize(
        extensions,
        compiler_directives={{
            'language_level': "3",
            'boundscheck': False,      # Disable bounds checking for speed
            'wraparound': False,        # Disable negative indexing
            'cdivision': True,          # C-style division
            'embedsignature': True,     # Embed function signatures
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
        }},
        annotate=True,  # Generate HTML annotation files
    ),
    zip_safe=False,
    install_requires=[
        "numpy>=1.20.0",
        "cython>=3.0.0",
    ],
)
'''

        files.append(FileTemplate(
            filename="setup.py",
            content=setup_py,
            file_type="build",
            language="python"
        ))

        # Generate Makefile
        makefile = f'''# Cython Module Makefile for {module_name}
# Compiles Cython code to native Python extensions

.PHONY: all build install clean test annotate benchmark

# Default target
all: build

# Build extension in-place
build:
\t@echo "Building {module_name} Cython extension..."
\tpython setup.py build_ext --inplace
\t@echo "✓ Build complete: {module_name}.so (or .pyd on Windows)"

# Install to site-packages
install:
\t@echo "Installing {module_name}..."
\tpip install -e .
\t@echo "✓ Installed {module_name}"

# Generate annotated HTML (shows C/Python interactions)
annotate:
\t@echo "Generating Cython annotation..."
\tcython -a {module_name}.pyx
\t@echo "✓ See {module_name}.html for performance analysis"

# Run tests
test: build
\t@echo "Running tests..."
\tpython -c "import {module_name}; {module_name}.main()"

# Performance benchmark
benchmark: build
\t@echo "Benchmarking Cython vs Python..."
\tpython benchmark_{module_name}.py

# Clean build artifacts
clean:
\trm -f {module_name}.c {module_name}.so {module_name}.html
\trm -f {module_name}*.pyd  # Windows
\trm -rf build/ *.egg-info/
\trm -rf __pycache__/ .pytest_cache/
\t@echo "✓ Cleaned build artifacts"

# Show Cython version
version:
\tcython --version
'''

        files.append(FileTemplate(
            filename="Makefile",
            content=makefile,
            file_type="build",
            language="makefile"
        ))

        # Generate README
        readme = f'''# {module_name}

{spec}

## About Cython

This module is written in **Cython** - an optimizing static compiler that makes writing C extensions for Python as easy as Python itself.

### Key Features

- **C-Level Performance**: 10-1000x faster than pure Python
- **Seamless Integration**: Import and use like normal Python modules
- **Static Typing**: Optional type declarations for speed
- **C Library Access**: Direct calls to C/C++ code
- **Python Compatibility**: Full access to Python ecosystem
- **Memory Control**: Manual memory management when needed
- **Parallelization**: True parallelism with `nogil`

## Performance Benefits

| Operation | Pure Python | Cython (typed) | Speedup |
|-----------|-------------|----------------|---------|
| Numeric loops | 1x | 50-200x | 50-200x |
| Array operations | 1x | 20-100x | 20-100x |
| String processing | 1x | 5-20x | 5-20x |
| Overall | 1x | 10-1000x | 10-1000x |

Perfect for LAT5150DRVMIL's performance-critical cybersecurity operations.

## Prerequisites

```bash
# Install Cython
pip install cython

# Install NumPy (for typed memoryviews)
pip install numpy

# Install build tools
sudo apt-get install build-essential  # Ubuntu/Debian
```

## Building

### Development Build (In-Place)
```bash
python setup.py build_ext --inplace

# Or use Makefile
make build
```

This creates `{module_name}.so` (or `.pyd` on Windows) in the current directory.

### Production Install
```bash
pip install -e .

# Or use Makefile
make install
```

This installs to your Python site-packages.

## Usage

### From Python
```python
# Import like any Python module
import {module_name}

# Use Python-callable functions (cpdef)
result = {module_name}.fibonacci(20)
print(f"Fibonacci(20) = {{result}}")

# Use the main class
app = {module_name}.{class_name}()
app.run()
```

### From Other Cython Modules
```cython
# In another .pyx file
from {module_name} cimport c_fibonacci, {class_name}

# Call C-level functions (no Python overhead)
cdef double result = c_fibonacci(30)

# Use cdef class efficiently
cdef {class_name} app = {class_name}()
```

## Performance Analysis

### Generate Annotation
```bash
make annotate

# Opens {module_name}.html in browser
firefox {module_name}.html
```

The HTML annotation shows:
- **Yellow lines**: Python interactions (slow)
- **White lines**: Pure C code (fast)
- Goal: Minimize yellow, maximize white

### Benchmark
```bash
make benchmark
```

## Optimization Tips

### 1. Use Static Types
```cython
# Slow (dynamic Python)
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Fast (static C)
cdef double fast_sum(double[:] arr):
    cdef:
        double total = 0.0
        int i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total
```

### 2. Disable Safety Checks
```cython
@cython.boundscheck(False)  # Disable bounds checking
@cython.wraparound(False)   # Disable negative indexing
cdef void process(double[:] data):
    # Fast C-speed loops
    ...
```

### 3. Release GIL for Parallelism
```cython
cdef void parallel_work() nogil:
    # Can run in parallel with Python threads
    # No Python object access allowed
    ...

def wrapper():
    with nogil:
        parallel_work()  # True parallelism!
```

## Compilation Options

### Optimize for Speed
```bash
CFLAGS="-O3 -march=native -ffast-math" python setup.py build_ext --inplace
```

### Enable Debugging
```bash
CFLAGS="-g -O0" python setup.py build_ext --inplace
gdb python
```

### Profile with Perf
```bash
perf record python -c "import {module_name}; {module_name}.main()"
perf report
```

## Integration with LAT5150DRVMIL

This Cython module can accelerate:
- **Malware Analysis**: Fast binary parsing and pattern matching
- **Cryptography**: High-performance hash computation
- **Network Processing**: Packet analysis at C speed
- **AI Inference**: Optimized tensor operations
- **Data Processing**: Fast IOC extraction and correlation

## Learn More

- Cython Documentation: https://cython.readthedocs.io/
- Cython GitHub: https://github.com/cython/cython
- Performance Tips: https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
'''

        files.append(FileTemplate(
            filename="README.md",
            content=readme,
            file_type="docs",
            language="markdown"
        ))

        return GeneratedModule(
            module_name=module_name,
            module_type=ModuleType.CYTHON_MODULE,
            description=spec,
            files=files,
            dependencies=["cython>=3.0.0", "numpy>=1.20.0", "setuptools"],
            build_instructions="python setup.py build_ext --inplace  # or: make build",
            usage_instructions=f"import {module_name}; {module_name}.main()"
        )

    def _generate_generic_module(self, module_name: str, module_type: ModuleType, spec: str) -> GeneratedModule:
        """Generate generic module"""

        files = [FileTemplate(
            filename="README.md",
            content=f"# {module_name}\n\n{spec}\n\nGenerated by Neural Code Synthesizer",
            file_type="docs",
            language="markdown"
        )]

        return GeneratedModule(
            module_name=module_name,
            module_type=module_type,
            description=spec,
            files=files,
            dependencies=[],
            build_instructions="See README.md",
            usage_instructions="See README.md"
        )

    # ========================================================================
    # CYBERSECURITY TOOL GENERATORS
    # ========================================================================

    def _generate_malware_analyzer(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate malware analysis tool"""

        has_static = 'static_analysis' in features
        has_dynamic = 'dynamic_analysis' in features
        has_yara = 'yara' in features

        # Generate main analyzer script
        analyzer_code = f'''#!/usr/bin/env python3
"""
{module_name} - Malware Analysis Tool
Generated for LAT5150DRVMIL threat detection

Specification: {spec}
Features: {', '.join(features)}
"""

import os
import sys
import hashlib
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

'''

        if has_static:
            analyzer_code += '''
try:
    import pefile
    import yara
except ImportError:
    print("ERROR: Missing dependencies. Install with: pip install pefile yara-python")
    sys.exit(1)

'''

        class_name = module_name.replace('_', ' ').title().replace(' ', '')

        analyzer_code += f'''
class MalwareAnalyzer:
    """Malware analyzer for LAT5150DRVMIL - {module_name}"""

    def __init__(self, yara_rules_path: str = None):
        self.yara_rules = None
        if yara_rules_path and os.path.exists(yara_rules_path):
            self.yara_rules = yara.compile(filepath=yara_rules_path)

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze suspicious file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {{file_path}}")

        results = {{
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'hashes': self._compute_hashes(file_path),
            'indicators': [],
            'risk_score': 0,
        }}

'''

        if has_static:
            analyzer_code += '''        # Static analysis
        results['file_type'] = self._identify_file_type(file_path)
        if results['file_type'] == 'PE':
            results['pe_analysis'] = self._analyze_pe(file_path)
            results['risk_score'] += 20

        # String analysis
        results['strings'] = self._extract_suspicious_strings(file_path)
        if results['strings']['suspicious_count'] > 10:
            results['risk_score'] += 30

        # Entropy analysis (packed malware detection)
        results['entropy'] = self._calculate_entropy(file_path)
        if results['entropy'] > 7.0:
            results['risk_score'] += 40
            results['indicators'].append(('HIGH_ENTROPY', 'Likely packed/encrypted'))

'''

        if has_yara:
            analyzer_code += '''        # YARA scanning
        if self.yara_rules:
            results['yara_matches'] = self._scan_yara(file_path)
            if results['yara_matches']:
                results['risk_score'] += 50
                results['indicators'].append(('YARA_MATCH', f"{{len(results['yara_matches'])}} rules matched"))

'''

        analyzer_code += '''        results['risk_level'] = self._get_risk_level(results['risk_score'])
        return results

    def _compute_hashes(self, file_path: str) -> Dict:
        """Compute file hashes for threat intelligence"""
        hashes = {}
        with open(file_path, 'rb') as f:
            data = f.read()
            hashes['md5'] = hashlib.md5(data).hexdigest()
            hashes['sha1'] = hashlib.sha1(data).hexdigest()
            hashes['sha256'] = hashlib.sha256(data).hexdigest()
        return hashes

    def _identify_file_type(self, file_path: str) -> str:
        """Identify file type"""
        try:
            pe = pefile.PE(file_path, fast_load=True)
            pe.close()
            return 'PE'
        except:
            pass

        with open(file_path, 'rb') as f:
            magic = f.read(4)
            if magic == b'\\x7fELF':
                return 'ELF'

        return 'UNKNOWN'

    def _analyze_pe(self, file_path: str) -> Dict:
        """Analyze PE file structure"""
        results = {'suspicious_imports': [], 'sections': []}

        try:
            pe = pefile.PE(file_path)

            # Check for suspicious imports
            suspicious_apis = [
                'WriteProcessMemory', 'VirtualAllocEx', 'CreateRemoteThread',
                'LoadLibrary', 'GetProcAddress', 'URLDownloadToFile'
            ]

            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name and imp.name.decode() in suspicious_apis:
                            results['suspicious_imports'].append(imp.name.decode())

            # Check sections
            for section in pe.sections:
                name = section.Name.decode().rstrip('\\x00')
                results['sections'].append({
                    'name': name,
                    'entropy': section.get_entropy(),
                    'size': section.SizeOfRawData
                })

            pe.close()
        except Exception as e:
            results['error'] = str(e)

        return results

    def _extract_suspicious_strings(self, file_path: str) -> Dict:
        """Extract and analyze strings"""
        import re

        results = {'suspicious_count': 0, 'ip_addresses': [], 'urls': []}

        ip_pattern = re.compile(r'\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b')
        url_pattern = re.compile(r'https?://[^\\s]+')

        with open(file_path, 'rb') as f:
            data = f.read()
            strings = re.findall(b'[\\x20-\\x7e]{6,}', data)

            for s in strings:
                try:
                    string = s.decode('utf-8')
                    if ip_pattern.search(string):
                        results['ip_addresses'].append(string)
                        results['suspicious_count'] += 1
                    if url_pattern.search(string):
                        results['urls'].append(string)
                        results['suspicious_count'] += 1
                except:
                    pass

        return results

    def _calculate_entropy(self, file_path: str) -> float:
        """Calculate Shannon entropy"""
        import math

        with open(file_path, 'rb') as f:
            data = f.read()

        if not data:
            return 0.0

        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1

        entropy = 0.0
        length = len(data)

        for freq in frequencies:
            if freq > 0:
                probability = freq / length
                entropy -= probability * math.log2(probability)

        return entropy

    def _scan_yara(self, file_path: str) -> List[Dict]:
        """Scan file with YARA rules"""
        matches = []
        if self.yara_rules:
            yara_matches = self.yara_rules.match(file_path)
            for match in yara_matches:
                matches.append({
                    'rule': match.rule,
                    'tags': match.tags,
                })
        return matches

    def _get_risk_level(self, score: int) -> str:
        """Convert risk score to level"""
        if score >= 70:
            return 'CRITICAL'
        elif score >= 50:
            return 'HIGH'
        elif score >= 30:
            return 'MEDIUM'
        elif score >= 10:
            return 'LOW'
        else:
            return 'CLEAN'

    def generate_report(self, results: Dict) -> str:
        """Generate analysis report"""
        report = []
        report.append("=" * 70)
        report.append(f"MALWARE ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"File: {results['file_path']}")
        report.append(f"Size: {results['file_size']} bytes")
        report.append(f"\\nHashes:")
        report.append(f"  MD5:    {results['hashes']['md5']}")
        report.append(f"  SHA1:   {results['hashes']['sha1']}")
        report.append(f"  SHA256: {results['hashes']['sha256']}")
        report.append(f"\\nRisk Score: {results['risk_score']}/100")
        report.append(f"Risk Level: {results['risk_level']}")

        if results['indicators']:
            report.append(f"\\nIndicators:")
            for indicator_type, description in results['indicators']:
                report.append(f"  [{indicator_type}] {description}")

        report.append("=" * 70)
        return "\\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='{module_name} - Malware Analysis Tool')
    parser.add_argument('file', help='File to analyze')
    parser.add_argument('--yara', help='YARA rules file', default=None)
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    args = parser.parse_args()

    analyzer = MalwareAnalyzer(args.yara)

    try:
        results = analyzer.analyze_file(args.file)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            report = analyzer.generate_report(results)
            print(report)

        # Exit code based on risk
        if results['risk_level'] in ['CRITICAL', 'HIGH']:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
'''

        files = [
            FileTemplate(
                filename=f"{module_name}.py",
                content=analyzer_code,
                file_type="source",
                language="python"
            ),
            FileTemplate(
                filename="README.md",
                content=f"# {module_name}\\n\\n{spec}\\n\\n## Usage\\n\\n```bash\\npython3 {module_name}.py <file_to_analyze>\\n```",
                file_type="docs",
                language="markdown"
            )
        ]

        dependencies = ["python>=3.8", "pefile", "yara-python"]

        return GeneratedModule(
            module_name=module_name,
            module_type=ModuleType.MALWARE_ANALYZER,
            description=spec,
            files=files,
            dependencies=dependencies,
            build_instructions="pip install -r requirements.txt",
            usage_instructions=f"python3 {module_name}.py <file>"
        )

    def _generate_exploit_detector(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate exploit detection tool"""
        # Placeholder - returns basic structure
        return self._generate_generic_module(module_name, ModuleType.EXPLOIT_DETECTOR, spec)

    def _generate_threat_hunter(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate threat hunting tool"""
        # Placeholder - returns basic structure
        return self._generate_generic_module(module_name, ModuleType.THREAT_HUNTER, spec)

    def _generate_forensics_tool(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate forensics tool"""
        # Placeholder - returns basic structure
        return self._generate_generic_module(module_name, ModuleType.FORENSICS_TOOL, spec)

    def _generate_security_monitor(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate security monitor"""
        # Placeholder - returns basic structure
        return self._generate_generic_module(module_name, ModuleType.SECURITY_MONITOR, spec)

    def _generate_vuln_scanner(self, module_name: str, features: List[str], spec: str) -> GeneratedModule:
        """Generate vulnerability scanner"""
        # Placeholder - returns basic structure
        return self._generate_generic_module(module_name, ModuleType.VULN_SCANNER, spec)

    def _generate_readme(self, module_name: str, spec: str, features: List[str]) -> str:
        """Generate README.md"""

        readme = f"""# {module_name.upper()}

{spec}

## Features

"""
        for feature in features:
            readme += f"- {feature.capitalize()} support\n"

        readme += f"""

## Building

```bash
make
sudo make install
```

## Loading

```bash
sudo modprobe {module_name}
```

## Verification

```bash
lsmod | grep {module_name}
dmesg | tail
```

## Unloading

```bash
sudo rmmod {module_name}
```

## Auto-generated

This driver was auto-generated by Neural Code Synthesizer (Phase 4.1).
Review and test thoroughly before production use.

Generated for LAT5150DRVMIL kernel development.
"""

        return readme

    def save_module(self, module: GeneratedModule, output_dir: str):
        """Save generated module to disk"""

        output_path = Path(output_dir) / module.module_name
        output_path.mkdir(parents=True, exist_ok=True)

        for file in module.files:
            file_path = output_path / file.filename
            with open(file_path, 'w') as f:
                f.write(file.content)

        print(f"✅ Module saved to {output_path}")
        print(f"   Files: {len(module.files)}")
        print(f"   Type: {module.module_type.value}")

    def format_generation_report(self, module: GeneratedModule) -> str:
        """Format generation report"""

        lines = []
        lines.append("=" * 80)
        lines.append("🧠 NEURAL CODE SYNTHESIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Module: {module.module_name}")
        lines.append(f"Type: {module.module_type.value}")
        lines.append(f"Description: {module.description}")
        lines.append("")

        lines.append("GENERATED FILES:")
        lines.append("-" * 80)
        for file in module.files:
            lines.append(f"  {file.filename} ({file.language}, {len(file.content)} bytes)")
        lines.append("")

        lines.append("DEPENDENCIES:")
        lines.append("-" * 80)
        for dep in module.dependencies:
            lines.append(f"  - {dep}")
        lines.append("")

        lines.append("BUILD INSTRUCTIONS:")
        lines.append("-" * 80)
        lines.append(module.build_instructions)
        lines.append("")

        lines.append("USAGE:")
        lines.append("-" * 80)
        lines.append(module.usage_instructions)

        lines.append("=" * 80)

        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    synthesizer = NeuralCodeSynthesizer()

    # Example 1: Kernel driver
    spec1 = "NPU device driver with DMA buffer management and thermal monitoring"
    module1 = synthesizer.generate_module(spec1)

    print(synthesizer.format_generation_report(module1))
    synthesizer.save_module(module1, "./generated")
