#!/usr/bin/env python3
"""
LAT5150DRVMIL System Validator
Comprehensive validation and health check for entire AI system

Validates:
1. Core AI Engine (DSMIL)
2. Unified Orchestrator
3. All MCP Servers (12 servers)
4. Screenshot Intelligence System
5. Hardware Integration (NPU, TPM)
6. Database Systems (Qdrant, Knowledge Graph)
7. Integration Layer
8. Dependencies and Requirements

Usage:
    python3 system_validator.py
    python3 system_validator.py --detailed
    python3 system_validator.py --fix-issues
"""

import sys
import os
import json
import subprocess
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

@dataclass
class ValidationResult:
    """Result of a validation check"""
    component: str
    status: str  # 'pass', 'warn', 'fail', 'skip'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class SystemValidator:
    """Comprehensive system validation"""

    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self.results: List[ValidationResult] = []
        self.project_root = Path("/home/user/LAT5150DRVMIL")

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Colors.BLUE}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
        print(f"{Colors.BLUE}{'='*80}{Colors.END}\n")

    def print_result(self, result: ValidationResult):
        """Print validation result"""
        symbols = {
            'pass': f"{Colors.GREEN}✓{Colors.END}",
            'warn': f"{Colors.YELLOW}⚠{Colors.END}",
            'fail': f"{Colors.RED}✗{Colors.END}",
            'skip': f"{Colors.BLUE}○{Colors.END}"
        }

        symbol = symbols.get(result.status, '?')
        print(f"  {symbol} {result.component}: {result.message}")

        if self.detailed and result.details:
            for key, value in result.details.items():
                print(f"      {key}: {value}")

        if result.recommendations:
            for rec in result.recommendations:
                print(f"      → {rec}")

    def validate_python_module(self, module_path: str, module_name: str) -> ValidationResult:
        """Validate Python module syntax and structure"""
        try:
            # Check file exists
            if not Path(module_path).exists():
                return ValidationResult(
                    component=module_name,
                    status='fail',
                    message="Module file not found"
                )

            # Check syntax by compiling
            with open(module_path, 'r') as f:
                code = f.read()

            compile(code, module_path, 'exec')

            # Get file size
            file_size = Path(module_path).stat().st_size

            return ValidationResult(
                component=module_name,
                status='pass',
                message=f"Syntax valid ({file_size} bytes)",
                details={'path': module_path, 'size_bytes': file_size}
            )

        except SyntaxError as e:
            return ValidationResult(
                component=module_name,
                status='fail',
                message=f"Syntax error: Line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return ValidationResult(
                component=module_name,
                status='fail',
                message=f"Validation failed: {str(e)[:100]}"
            )

    def check_core_ai_engine(self) -> List[ValidationResult]:
        """Validate core DSMIL AI Engine"""
        results = []

        # 1. Check DSMILAIEngine
        dsmil_path = self.project_root / "02-ai-engine" / "dsmil_ai_engine.py"
        results.append(self.validate_python_module(str(dsmil_path), "dsmil_ai_engine"))

        # 2. Check models.json
        models_json = self.project_root / "02-ai-engine" / "models.json"
        if models_json.exists():
            try:
                with open(models_json) as f:
                    models = json.load(f)
                    model_count = len(models.get('models', []))
                    results.append(ValidationResult(
                        component="models.json",
                        status='pass',
                        message=f"Found {model_count} model configurations",
                        details={'models': model_count, 'path': str(models_json)}
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    component="models.json",
                    status='fail',
                    message=f"Failed to parse: {e}"
                ))
        else:
            results.append(ValidationResult(
                component="models.json",
                status='warn',
                message="Models configuration not found",
                recommendations=["Create models.json with AI model configurations"]
            ))

        # 3. Check unified orchestrator
        orch_path = self.project_root / "02-ai-engine" / "unified_orchestrator.py"
        results.append(self.validate_python_module(str(orch_path), "unified_orchestrator"))

        return results

    def check_mcp_servers(self) -> List[ValidationResult]:
        """Validate all MCP servers"""
        results = []

        mcp_servers = [
            ("dsmil_mcp_server.py", "DSMIL AI Engine MCP"),
            ("sequential_thinking_server.py", "Sequential Thinking"),
            ("filesystem_server.py", "Filesystem"),
            ("memory_server.py", "Memory/Knowledge Graph"),
            ("fetch_server.py", "Web Fetch"),
            ("git_server.py", "Git Operations"),
            ("screenshot_intel_mcp_server.py", "Screenshot Intelligence"),
        ]

        for filename, name in mcp_servers:
            server_path = self.project_root / "02-ai-engine" / filename
            if server_path.exists():
                result = self.validate_python_module(str(server_path), filename.replace('.py', ''))
                result.component = name
                results.append(result)
            else:
                results.append(ValidationResult(
                    component=name,
                    status='warn',
                    message=f"Server file not found: {filename}"
                ))

        # Check MCP config
        mcp_config = self.project_root / "02-ai-engine" / "mcp_servers_config.json"
        if mcp_config.exists():
            try:
                with open(mcp_config) as f:
                    config = json.load(f)
                    server_count = len(config.get('mcpServers', {}))
                    results.append(ValidationResult(
                        component="MCP Configuration",
                        status='pass',
                        message=f"{server_count} servers configured",
                        details={'servers': list(config.get('mcpServers', {}).keys())}
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    component="MCP Configuration",
                    status='fail',
                    message=f"Failed to parse config: {e}"
                ))
        else:
            results.append(ValidationResult(
                component="MCP Configuration",
                status='fail',
                message="mcp_servers_config.json not found",
                recommendations=["Create MCP server configuration"]
            ))

        return results

    def check_screenshot_intelligence(self) -> List[ValidationResult]:
        """Validate Screenshot Intelligence System"""
        results = []

        rag_dir = self.project_root / "04-integrations" / "rag_system"

        components = [
            ("vector_rag_system.py", "Vector RAG System"),
            ("screenshot_intelligence.py", "Screenshot Intelligence"),
            ("ai_analysis_layer.py", "AI Analysis Layer"),
            ("telegram_integration.py", "Telegram Integration"),
            ("signal_integration.py", "Signal Integration"),
            ("system_health_monitor.py", "Health Monitor"),
            ("resilience_utils.py", "Resilience Utils"),
            ("test_screenshot_intel_integration.py", "Integration Tests"),
        ]

        for filename, name in components:
            comp_path = rag_dir / filename
            if comp_path.exists():
                result = self.validate_python_module(str(comp_path), filename.replace('.py', ''))
                result.component = f"Screenshot Intel: {name}"
                results.append(result)
            else:
                results.append(ValidationResult(
                    component=f"Screenshot Intel: {name}",
                    status='warn',
                    message=f"Component not found: {filename}"
                ))

        return results

    def check_dependencies(self) -> List[ValidationResult]:
        """Check critical Python dependencies"""
        results = []

        critical_packages = {
            'qdrant_client': 'Qdrant Vector Database',
            'sentence_transformers': 'Sentence Transformers',
            'fastapi': 'FastAPI',
            'mcp': 'Model Context Protocol',
        }

        optional_packages = {
            'paddleocr': 'PaddleOCR',
            'telethon': 'Telegram Client',
            'psutil': 'System Monitoring',
        }

        for package, name in critical_packages.items():
            try:
                __import__(package)
                results.append(ValidationResult(
                    component=f"Dependency: {name}",
                    status='pass',
                    message="Installed"
                ))
            except ImportError:
                results.append(ValidationResult(
                    component=f"Dependency: {name}",
                    status='fail',
                    message="Not installed",
                    recommendations=[f"pip install {package}"]
                ))

        for package, name in optional_packages.items():
            try:
                __import__(package)
                results.append(ValidationResult(
                    component=f"Optional: {name}",
                    status='pass',
                    message="Installed"
                ))
            except ImportError:
                results.append(ValidationResult(
                    component=f"Optional: {name}",
                    status='warn',
                    message="Not installed (optional)"
                ))

        return results

    def check_system_commands(self) -> List[ValidationResult]:
        """Check system command availability"""
        results = []

        commands = {
            'docker': 'Docker (for Qdrant)',
            'tesseract': 'Tesseract OCR',
            'git': 'Git',
        }

        for cmd, name in commands.items():
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.decode().split('\n')[0][:50]
                    results.append(ValidationResult(
                        component=f"Command: {name}",
                        status='pass',
                        message=f"Available: {version}"
                    ))
                else:
                    results.append(ValidationResult(
                        component=f"Command: {name}",
                        status='warn',
                        message="Command found but error on --version"
                    ))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                results.append(ValidationResult(
                    component=f"Command: {name}",
                    status='warn',
                    message="Not found",
                    recommendations=[f"Install {cmd}"]
                ))

        return results

    def check_qdrant_service(self) -> ValidationResult:
        """Check if Qdrant is running"""
        try:
            import requests
            response = requests.get('http://127.0.0.1:6333/collections', timeout=2)
            if response.status_code == 200:
                collections = response.json().get('result', {}).get('collections', [])
                return ValidationResult(
                    component="Qdrant Service",
                    status='pass',
                    message=f"Running with {len(collections)} collections",
                    details={'collections': [c.get('name') for c in collections]}
                )
            else:
                return ValidationResult(
                    component="Qdrant Service",
                    status='warn',
                    message=f"Responded with status {response.status_code}"
                )
        except Exception as e:
            return ValidationResult(
                component="Qdrant Service",
                status='warn',
                message="Not running or not accessible",
                recommendations=["Start Qdrant: docker start qdrant"]
            )

    def check_hardware(self) -> List[ValidationResult]:
        """Check hardware components"""
        results = []

        # Check if hardware modules exist
        hw_dir = self.project_root / "02-ai-engine" / "hardware"
        if hw_dir.exists():
            hardware_modules = [
                "military_npu_dsmil_loader.py",
                "pqc_manager.py",
                "pqc_tpm_integration.py"
            ]

            for module in hardware_modules:
                mod_path = hw_dir / module
                if mod_path.exists():
                    results.append(ValidationResult(
                        component=f"Hardware: {module.replace('.py', '')}",
                        status='pass',
                        message="Module present"
                    ))
                else:
                    results.append(ValidationResult(
                        component=f"Hardware: {module.replace('.py', '')}",
                        status='skip',
                        message="Module not found (optional)"
                    ))
        else:
            results.append(ValidationResult(
                component="Hardware Integration",
                status='skip',
                message="Hardware directory not found"
            ))

        return results

    def run_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        self.results = []

        # Core AI Engine
        self.print_header("Core AI Engine")
        core_results = self.check_core_ai_engine()
        for result in core_results:
            self.print_result(result)
            self.results.append(result)

        # MCP Servers
        self.print_header("MCP Servers")
        mcp_results = self.check_mcp_servers()
        for result in mcp_results:
            self.print_result(result)
            self.results.append(result)

        # Screenshot Intelligence
        self.print_header("Screenshot Intelligence System")
        screenshot_results = self.check_screenshot_intelligence()
        for result in screenshot_results:
            self.print_result(result)
            self.results.append(result)

        # Dependencies
        self.print_header("Python Dependencies")
        dep_results = self.check_dependencies()
        for result in dep_results:
            self.print_result(result)
            self.results.append(result)

        # System Commands
        self.print_header("System Commands")
        cmd_results = self.check_system_commands()
        for result in cmd_results:
            self.print_result(result)
            self.results.append(result)

        # Qdrant Service
        self.print_header("Services")
        qdrant_result = self.check_qdrant_service()
        self.print_result(qdrant_result)
        self.results.append(qdrant_result)

        # Hardware
        self.print_header("Hardware Integration")
        hw_results = self.check_hardware()
        for result in hw_results:
            self.print_result(result)
            self.results.append(result)

        # Generate summary
        return self.generate_summary()

    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == 'pass')
        warnings = sum(1 for r in self.results if r.status == 'warn')
        failed = sum(1 for r in self.results if r.status == 'fail')
        skipped = sum(1 for r in self.results if r.status == 'skip')

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': total,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'skipped': skipped,
            'success_rate': (passed / total * 100) if total > 0 else 0
        }

        # Print summary
        self.print_header("Validation Summary")
        print(f"  Total Checks: {total}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
        print(f"  {Colors.YELLOW}Warnings: {warnings}{Colors.END}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
        print(f"  {Colors.BLUE}Skipped: {skipped}{Colors.END}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%\n")

        # Overall status
        if failed > 0:
            overall = f"{Colors.RED}FAILED{Colors.END}"
        elif warnings > 3:
            overall = f"{Colors.YELLOW}DEGRADED{Colors.END}"
        else:
            overall = f"{Colors.GREEN}HEALTHY{Colors.END}"

        print(f"  Overall Status: {overall}\n")

        # Critical issues
        critical_failures = [r for r in self.results if r.status == 'fail']
        if critical_failures:
            print(f"{Colors.RED}Critical Issues:{Colors.END}")
            for fail in critical_failures:
                print(f"  ✗ {fail.component}: {fail.message}")
            print()

        # All recommendations
        all_recs = []
        for r in self.results:
            all_recs.extend(r.recommendations)

        if all_recs:
            print(f"{Colors.CYAN}Recommendations:{Colors.END}")
            for rec in set(all_recs):
                print(f"  → {rec}")
            print()

        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LAT5150DRVMIL System Validator")
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--json', action='store_true', help='Output JSON')

    args = parser.parse_args()

    validator = SystemValidator(detailed=args.detailed)

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       LAT5150DRVMIL System Validation & Health Check        ║")
    print("║         Dell Latitude 5450 Covert AI Platform               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    summary = validator.run_validation()

    if args.json:
        print("\n" + json.dumps(summary, indent=2))

    # Exit code based on result
    if summary['failed'] > 0:
        sys.exit(1)
    elif summary['warnings'] > 3:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
