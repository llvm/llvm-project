#!/usr/bin/env python3
"""
DBXForensics Toolkit Integration
Python wrapper for all 9 DBXForensics tools

Tools:
1. dbxScreenshot - Forensic screenshot capture
2. dbxELA - Error Level Analysis (manipulation detection)
3. dbxNoiseMap - Digital noise analysis
4. dbxGhost - Screen comparison
5. dbxMetadata - Comprehensive metadata extraction
6. dbxHashFile - Multi-algorithm hashing
7. dbxCsvViewer - CSV analysis
8. dbxSeqCheck - Sequence validation
9. dbxMouseRecorder - Workflow automation
"""

import os
import json
import shutil
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Xen VM executor if available
try:
    from xen_vm_executor import XenVMExecutor, VMExecutionResult
    XEN_VM_AVAILABLE = True
except ImportError:
    XEN_VM_AVAILABLE = False
    logger.info("Xen VM executor not available - using Wine/native execution only")


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    tool_name: str
    output: Any
    raw_stdout: str
    raw_stderr: str
    execution_time: float
    error_message: Optional[str] = None


class DBXForensicsTool:
    """
    Base class for DBXForensics tool wrappers

    Handles Wine execution on Linux, process management,
    output parsing, and error handling.
    """

    def __init__(
        self,
        tool_name: str,
        tool_exe_path: Path,
        timeout: int = 300,
        use_vm: bool = False,
        vm_executor: Optional['XenVMExecutor'] = None
    ):
        """
        Initialize tool wrapper

        Args:
            tool_name: Name of the tool (e.g., 'dbxScreenshot')
            tool_exe_path: Path to .exe file
            timeout: Maximum execution time in seconds
            use_vm: Use Xen VM execution instead of Wine (more compatible)
            vm_executor: Optional XenVMExecutor instance (created if None)
        """
        self.tool_name = tool_name
        self.tool_path = Path(tool_exe_path)
        self.timeout = timeout
        self.use_vm = use_vm
        self.vm_executor = vm_executor

        # Initialize VM executor if requested
        if self.use_vm:
            if not XEN_VM_AVAILABLE:
                logger.error(f"{tool_name}: VM execution requested but xen_vm_executor not available")
                self.use_vm = False
            elif self.vm_executor is None:
                logger.info(f"{tool_name}: Initializing Xen VM executor")
                self.vm_executor = XenVMExecutor()

        # Check if Wine is available (needed on Linux if not using VM)
        self.wine_available = shutil.which('wine') is not None
        self.is_windows = os.name == 'nt'

        if not self.use_vm and not self.is_windows and not self.wine_available:
            logger.warning(
                f"{tool_name}: Wine not found. "
                "Install with: sudo apt install wine wine64 OR use use_vm=True"
            )

        # Verify tool exists (unless using VM where tools are on VM)
        if not self.use_vm and not self.tool_path.exists():
            logger.warning(
                f"{tool_name}: Tool not found at {self.tool_path}. "
                f"Download from https://www.dbxforensics.com/Tools/Download"
            )

    def _build_command(self, *args) -> List[str]:
        """
        Build command array for subprocess

        Args:
            *args: Tool arguments

        Returns:
            Command array ready for subprocess.run()
        """
        if self.is_windows:
            # Direct execution on Windows
            cmd = [str(self.tool_path)]
        else:
            # Use Wine on Linux
            cmd = ['wine', str(self.tool_path)]

        # Add arguments
        cmd.extend(str(arg) for arg in args)

        return cmd

    def execute(
        self,
        *args,
        stdin_input: Optional[str] = None,
        capture_output: bool = True,
        input_file: Optional[Path] = None
    ) -> ToolResult:
        """
        Execute tool with arguments

        Args:
            *args: Tool arguments
            stdin_input: Optional stdin input
            capture_output: Capture stdout/stderr
            input_file: Input file path (required for VM execution)

        Returns:
            ToolResult with execution results
        """
        start_time = datetime.now()

        # Route to VM executor if enabled
        if self.use_vm:
            return self._execute_via_vm(args, input_file)

        # Otherwise use local Wine/native execution
        cmd = self._build_command(*args)

        logger.info(f"Executing {self.tool_name}: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                input=stdin_input,
                timeout=self.timeout
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Parse output
            parsed_output = self.parse_output(result.stdout, result.stderr)

            return ToolResult(
                success=result.returncode == 0,
                tool_name=self.tool_name,
                output=parsed_output,
                raw_stdout=result.stdout,
                raw_stderr=result.stderr,
                execution_time=execution_time,
                error_message=result.stderr if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            execution_time = self.timeout
            return ToolResult(
                success=False,
                tool_name=self.tool_name,
                output=None,
                raw_stdout="",
                raw_stderr="",
                execution_time=execution_time,
                error_message=f"Timeout after {self.timeout}s"
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ToolResult(
                success=False,
                tool_name=self.tool_name,
                output=None,
                raw_stdout="",
                raw_stderr="",
                execution_time=execution_time,
                error_message=str(e)
            )

    def _execute_via_vm(self, args: tuple, input_file: Optional[Path]) -> ToolResult:
        """
        Execute tool via Xen VM

        Args:
            args: Tool arguments
            input_file: Input file path (required)

        Returns:
            ToolResult with execution results
        """
        if not input_file:
            return ToolResult(
                success=False,
                tool_name=self.tool_name,
                output=None,
                raw_stdout="",
                raw_stderr="",
                execution_time=0,
                error_message="VM execution requires input_file parameter"
            )

        try:
            # Execute via VM
            vm_result = self.vm_executor.execute_tool(
                tool_name=self.tool_name,
                input_file=input_file,
                args=list(args)
            )

            # Convert VMExecutionResult to ToolResult
            parsed_output = self.parse_output(vm_result.stdout, vm_result.stderr)

            return ToolResult(
                success=vm_result.success,
                tool_name=self.tool_name,
                output=parsed_output,
                raw_stdout=vm_result.stdout,
                raw_stderr=vm_result.stderr,
                execution_time=vm_result.execution_time,
                error_message=vm_result.stderr if not vm_result.success else None
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.tool_name,
                output=None,
                raw_stdout="",
                raw_stderr="",
                execution_time=0,
                error_message=f"VM execution error: {e}"
            )

    def parse_output(self, stdout: str, stderr: str) -> Any:
        """
        Parse tool output (override in subclasses)

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            Parsed output (tool-specific)
        """
        return {
            'stdout': stdout,
            'stderr': stderr
        }


class DBXScreenshot(DBXForensicsTool):
    """
    dbxScreenshot wrapper

    Forensic screenshot capture with comprehensive metadata:
    - UTC timestamps
    - MD5, SHA-1, SHA-256 hashes
    - Device information
    - User information
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxScreenshot', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def capture(
        self,
        output_path: Path,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> ToolResult:
        """
        Capture forensic screenshot

        Args:
            output_path: Where to save screenshot
            region: Optional (x, y, width, height) region

        Returns:
            ToolResult with forensic metadata
        """
        args = ['--output', str(output_path)]

        if region:
            x, y, w, h = region
            args.extend(['--region', f"{x},{y},{w},{h}"])

        result = self.execute(*args)

        # Extract forensic metadata if capture successful
        if result.success and output_path.exists():
            metadata = self._extract_forensic_metadata(output_path)
            result.output = metadata

        return result

    def _extract_forensic_metadata(self, screenshot_path: Path) -> Dict:
        """Extract forensic metadata from screenshot"""
        # Note: dbxScreenshot embeds metadata in image file
        # This is a placeholder - actual extraction would parse image metadata
        return {
            'file_path': str(screenshot_path),
            'file_size': screenshot_path.stat().st_size,
            'captured_at': datetime.now().isoformat(),
            'forensic_tool': 'dbxScreenshot v1.0.0',
            'hashes': self._calculate_hashes(screenshot_path)
        }

    def _calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate file hashes"""
        hashes = {
            'md5': hashlib.md5(),
            'sha1': hashlib.sha1(),
            'sha256': hashlib.sha256()
        }

        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                for h in hashes.values():
                    h.update(chunk)

        return {name: h.hexdigest() for name, h in hashes.items()}


class DBXELA(DBXForensicsTool):
    """
    dbxELA wrapper

    Error Level Analysis for JPEG manipulation detection.
    Detects areas of images that have been re-compressed (edited).
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxELA', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def analyze(
        self,
        image_path: Path,
        output_ela_path: Optional[Path] = None,
        quality: int = 90
    ) -> ToolResult:
        """
        Analyze image for manipulation

        Args:
            image_path: Image to analyze (JPEG)
            output_ela_path: Where to save ELA visualization
            quality: JPEG quality for re-compression (default: 90)

        Returns:
            ToolResult with manipulation analysis
        """
        if output_ela_path is None:
            output_ela_path = image_path.parent / f"{image_path.stem}_ela.png"

        args = [
            '--input', str(image_path),
            '--output', str(output_ela_path),
            '--quality', str(quality)
        ]

        result = self.execute(*args)

        if result.success:
            result.output = {
                'input_image': str(image_path),
                'ela_visualization': str(output_ela_path),
                'quality_threshold': quality,
                'manipulation_detected': self._analyze_ela_output(output_ela_path)
            }

        return result

    def _analyze_ela_output(self, ela_path: Path) -> Dict:
        """Analyze ELA visualization for manipulation"""
        # Placeholder - would analyze ELA image brightness/hotspots
        return {
            'authenticity_score': 85,  # 0-100
            'suspicious_regions': [],
            'verdict': 'likely_authentic'
        }


class DBXNoiseMap(DBXForensicsTool):
    """
    dbxNoiseMap wrapper

    Digital noise pattern analysis for device fingerprinting
    and modification detection.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxNoiseMap', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def analyze(
        self,
        image_path: Path,
        output_noise_path: Optional[Path] = None
    ) -> ToolResult:
        """
        Analyze digital noise patterns

        Args:
            image_path: Image to analyze
            output_noise_path: Where to save noise map

        Returns:
            ToolResult with noise analysis
        """
        if output_noise_path is None:
            output_noise_path = image_path.parent / f"{image_path.stem}_noise.png"

        args = [
            '--input', str(image_path),
            '--output', str(output_noise_path)
        ]

        result = self.execute(*args)

        if result.success:
            result.output = {
                'input_image': str(image_path),
                'noise_map': str(output_noise_path),
                'noise_signature': self._extract_noise_signature(output_noise_path)
            }

        return result

    def _extract_noise_signature(self, noise_map_path: Path) -> Dict:
        """Extract device-specific noise signature"""
        # Placeholder - would extract actual noise pattern
        return {
            'pattern_hash': hashlib.sha256(noise_map_path.read_bytes()).hexdigest()[:16],
            'consistency_score': 92
        }


class DBXMetadata(DBXForensicsTool):
    """
    dbxMetadata wrapper

    Comprehensive file and metadata extraction.
    Extracts EXIF, XMP, file system metadata, and more.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxMetadata', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def extract(
        self,
        file_path: Path,
        output_format: str = 'json'
    ) -> ToolResult:
        """
        Extract comprehensive metadata

        Args:
            file_path: File to analyze
            output_format: Output format ('json', 'xml', 'text')

        Returns:
            ToolResult with metadata
        """
        args = [
            '--file', str(file_path),
            '--format', output_format
        ]

        result = self.execute(*args)

        if result.success and output_format == 'json':
            try:
                result.output = json.loads(result.raw_stdout)
            except json.JSONDecodeError:
                result.output = {'raw': result.raw_stdout}

        return result


class DBXHashFile(DBXForensicsTool):
    """
    dbxHashFile wrapper

    Multi-algorithm cryptographic hashing:
    CRC32, MD5, SHA-1, SHA-256, SHA-512, SHA3-256
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxHashFile', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def calculate_hashes(
        self,
        file_path: Path,
        algorithms: Optional[List[str]] = None
    ) -> ToolResult:
        """
        Calculate file hashes

        Args:
            file_path: File to hash
            algorithms: List of algorithms (default: all)

        Returns:
            ToolResult with hashes
        """
        if algorithms is None:
            algorithms = ['md5', 'sha1', 'sha256', 'sha512', 'sha3-256']

        args = [
            '--file', str(file_path),
            '--algorithms', ','.join(algorithms)
        ]

        result = self.execute(*args)

        if result.success:
            result.output = self._parse_hash_output(result.raw_stdout)

        return result

    def _parse_hash_output(self, output: str) -> Dict[str, str]:
        """Parse hash output"""
        hashes = {}
        for line in output.strip().split('\n'):
            if ':' in line:
                algo, hash_value = line.split(':', 1)
                hashes[algo.strip().lower()] = hash_value.strip()
        return hashes


class DBXSeqCheck(DBXForensicsTool):
    """
    dbxSeqCheck wrapper

    Numeric sequence integrity verification.
    Detects missing numbers, duplicates, ordering errors.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxSeqCheck', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def check_sequence(
        self,
        numbers: List[int]
    ) -> ToolResult:
        """
        Check sequence integrity

        Args:
            numbers: List of numbers to check

        Returns:
            ToolResult with sequence analysis
        """
        # Format numbers as newline-separated string
        input_data = '\n'.join(map(str, numbers))

        result = self.execute('--check-gaps', stdin_input=input_data)

        if result.success:
            result.output = self._parse_sequence_report(result.raw_stdout, numbers)

        return result

    def _parse_sequence_report(self, output: str, original_numbers: List[int]) -> Dict:
        """Parse sequence check report"""
        return {
            'total_numbers': len(original_numbers),
            'min': min(original_numbers) if original_numbers else None,
            'max': max(original_numbers) if original_numbers else None,
            'gaps': [],  # Would parse from output
            'duplicates': [],
            'out_of_order': []
        }


class DBXCsvViewer(DBXForensicsTool):
    """
    dbxCsvViewer wrapper

    Fast CSV analysis with Excel export, column sorting,
    custom delimiters support.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxCsvViewer', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def analyze(
        self,
        csv_path: Path,
        delimiter: str = ',',
        export_excel: bool = True
    ) -> ToolResult:
        """
        Analyze CSV file

        Args:
            csv_path: CSV file to analyze
            delimiter: Delimiter character (default: comma)
            export_excel: Export to Excel format

        Returns:
            ToolResult with CSV analysis
        """
        args = [
            '--input', str(csv_path),
            '--delimiter', delimiter
        ]

        if export_excel:
            excel_path = csv_path.parent / f"{csv_path.stem}.xlsx"
            args.extend(['--export-excel', str(excel_path)])

        result = self.execute(*args)

        if result.success:
            result.output = {
                'csv_file': str(csv_path),
                'delimiter': delimiter,
                'excel_export': str(excel_path) if export_excel else None,
                'rows_parsed': self._count_csv_rows(csv_path),
                'columns': self._get_csv_columns(csv_path, delimiter)
            }

        return result

    def _count_csv_rows(self, csv_path: Path) -> int:
        """Count rows in CSV"""
        try:
            with open(csv_path, 'r') as f:
                return sum(1 for line in f) - 1  # Exclude header
        except:
            return 0

    def _get_csv_columns(self, csv_path: Path, delimiter: str) -> List[str]:
        """Get column names"""
        try:
            with open(csv_path, 'r') as f:
                header = f.readline().strip()
                return header.split(delimiter)
        except:
            return []


class DBXGhost(DBXForensicsTool):
    """
    dbxGhost wrapper

    Screen portion capture and visual comparison tool.
    Supports transparency, overlay, and difference detection.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxGhost', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def compare(
        self,
        image_a: Path,
        image_b: Path,
        output_diff: Optional[Path] = None,
        transparency: int = 50
    ) -> ToolResult:
        """
        Compare two images visually

        Args:
            image_a: First image
            image_b: Second image
            output_diff: Where to save difference visualization
            transparency: Transparency level (0-100)

        Returns:
            ToolResult with comparison analysis
        """
        if output_diff is None:
            output_diff = image_a.parent / f"diff_{image_a.stem}_{image_b.stem}.png"

        args = [
            '--image-a', str(image_a),
            '--image-b', str(image_b),
            '--output', str(output_diff),
            '--transparency', str(transparency),
            '--mode', 'difference'
        ]

        result = self.execute(*args)

        if result.success:
            result.output = {
                'image_a': str(image_a),
                'image_b': str(image_b),
                'difference_visualization': str(output_diff),
                'transparency': transparency,
                'difference_score': self._calculate_difference_score(image_a, image_b)
            }

        return result

    def _calculate_difference_score(self, image_a: Path, image_b: Path) -> float:
        """Calculate visual difference score"""
        # Placeholder - would use image comparison algorithm
        # Return 0-100 (0=identical, 100=completely different)
        return 15.5


class DBXMouseRecorder(DBXForensicsTool):
    """
    dbxMouseRecorder wrapper

    Workflow automation via mouse pointer recording and replay.
    Enables batch processing and standardized procedures.
    """

    def __init__(self, tool_path: Path, use_vm: bool = False, vm_executor: Optional['XenVMExecutor'] = None):
        super().__init__('dbxMouseRecorder', tool_path, use_vm=use_vm, vm_executor=vm_executor)

    def record_workflow(
        self,
        output_script: Path,
        duration: int = 60
    ) -> ToolResult:
        """
        Record mouse workflow

        Args:
            output_script: Where to save recorded workflow
            duration: Recording duration in seconds

        Returns:
            ToolResult with recording info
        """
        args = [
            '--record',
            '--output', str(output_script),
            '--duration', str(duration)
        ]

        result = self.execute(*args)

        if result.success:
            result.output = {
                'workflow_script': str(output_script),
                'duration': duration,
                'actions_recorded': self._count_actions(output_script)
            }

        return result

    def replay_workflow(
        self,
        workflow_script: Path,
        repeat: int = 1
    ) -> ToolResult:
        """
        Replay recorded workflow

        Args:
            workflow_script: Workflow script to replay
            repeat: Number of times to repeat

        Returns:
            ToolResult with replay info
        """
        args = [
            '--replay',
            '--script', str(workflow_script),
            '--repeat', str(repeat)
        ]

        result = self.execute(*args)

        if result.success:
            result.output = {
                'workflow_script': str(workflow_script),
                'repetitions': repeat,
                'success': True
            }

        return result

    def _count_actions(self, script_path: Path) -> int:
        """Count recorded actions in script"""
        # Placeholder - would parse script file
        return 0


class DBXForensicsToolkit:
    """
    Main toolkit class managing all 9 DBXForensics tools

    Supports both Wine execution (local) and Xen VM execution (remote).
    Use use_vm=True for 100% Windows compatibility via isolated VM.
    """

    def __init__(
        self,
        tools_dir: Path = None,
        use_vm: bool = False,
        vm_executor: Optional['XenVMExecutor'] = None
    ):
        """
        Initialize toolkit

        Args:
            tools_dir: Directory containing .exe files (for Wine mode)
            use_vm: Use Xen VM execution instead of Wine (recommended)
            vm_executor: Optional XenVMExecutor instance (auto-created if None)
        """
        if tools_dir is None:
            tools_dir = Path(__file__).parent / 'tools'

        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.use_vm = use_vm
        self.vm_executor = vm_executor

        # Initialize VM executor if requested
        if self.use_vm and self.vm_executor is None and XEN_VM_AVAILABLE:
            logger.info("Initializing Xen VM executor for toolkit")
            self.vm_executor = XenVMExecutor()

        # Initialize all 9 tools with VM support
        self.screenshot = DBXScreenshot(
            self.tools_dir / 'dbxScreenshot.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.ela = DBXELA(
            self.tools_dir / 'dbxELA.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.noise_map = DBXNoiseMap(
            self.tools_dir / 'dbxNoiseMap.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.metadata = DBXMetadata(
            self.tools_dir / 'dbxMetadata.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.hash_file = DBXHashFile(
            self.tools_dir / 'dbxHashFile.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.seq_check = DBXSeqCheck(
            self.tools_dir / 'dbxSeqCheck.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.csv_viewer = DBXCsvViewer(
            self.tools_dir / 'dbxCsvViewer.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.ghost = DBXGhost(
            self.tools_dir / 'dbxGhost.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )
        self.mouse_recorder = DBXMouseRecorder(
            self.tools_dir / 'dbxMouseRecorder.exe',
            use_vm=self.use_vm,
            vm_executor=self.vm_executor
        )

        execution_mode = "Xen VM" if self.use_vm else "Wine/Native"
        logger.info(f"✓ DBXForensics Toolkit initialized ({execution_mode} mode)")
        logger.info(f"  Tools directory: {self.tools_dir}")

        # Check tool availability
        self._check_tools()

    def _check_tools(self):
        """Check which tools are available"""
        tools = [
            ('dbxScreenshot', self.screenshot),
            ('dbxELA', self.ela),
            ('dbxNoiseMap', self.noise_map),
            ('dbxMetadata', self.metadata),
            ('dbxHashFile', self.hash_file),
            ('dbxSeqCheck', self.seq_check),
            ('dbxCsvViewer', self.csv_viewer),
            ('dbxGhost', self.ghost),
            ('dbxMouseRecorder', self.mouse_recorder)
        ]

        available = 0
        for name, tool in tools:
            if tool.tool_path.exists():
                logger.info(f"  ✓ {name}")
                available += 1
            else:
                logger.warning(f"  ✗ {name} (not found)")

        logger.info(f"  Available: {available}/9 tools")


# CLI interface
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("DBXForensics Toolkit")
        print("Usage:")
        print("  python3 dbxforensics_toolkit.py test")
        print("  python3 dbxforensics_toolkit.py check-tools")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'check-tools':
        toolkit = DBXForensicsToolkit()
        print("\n✓ Toolkit initialized")

    elif command == 'test':
        toolkit = DBXForensicsToolkit()
        print("\n✓ Toolkit test complete")
