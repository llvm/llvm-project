#!/usr/bin/env python3
"""
Xen VM Executor for DBXForensics Tools

Executes Windows forensics tools on isolated Xen VM via RPC.
Provides transparent interface - same API as Wine execution.

Usage:
    from xen_vm_executor import XenVMExecutor

    executor = XenVMExecutor(vm_ip="192.168.100.10")

    if executor.check_health():
        result = executor.execute_tool(
            tool_name="dbxELA",
            input_file=Path("screenshot.jpg"),
            args=["/quality:90"]
        )
        print(result['stdout'])
"""

import requests
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class VMExecutionResult:
    """Result from VM tool execution"""
    success: bool
    tool_name: str
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    vm_ip: str
    timestamp: str


class XenVMExecutor:
    """
    Execute DBXForensics tools on Xen Windows VM via RPC

    Architecture:
    - Dom0 (Linux) sends RPC requests to DomU (Windows VM)
    - VM runs Flask RPC server (forensics_rpc_server.py)
    - Files transferred via shared folder (SMB/CIFS mount)
    - Results returned as JSON

    Key Features:
    - Health checking (VM availability)
    - Automatic file transfer to shared folder
    - Timeout protection
    - Error handling and retry logic
    - Logging and diagnostics
    """

    def __init__(
        self,
        vm_ip: str = "192.168.100.10",
        vm_port: int = 5000,
        shared_input_dir: Path = None,
        shared_output_dir: Path = None,
        timeout: int = 300,
        retry_count: int = 3
    ):
        """
        Initialize VM executor

        Args:
            vm_ip: IP address of forensics Windows VM
            vm_port: RPC server port (default: 5000)
            shared_input_dir: Shared folder for input files
            shared_output_dir: Shared folder for output files
            timeout: Tool execution timeout (seconds)
            retry_count: Number of retries on failure
        """
        self.vm_ip = vm_ip
        self.vm_port = vm_port
        self.base_url = f"http://{vm_ip}:{vm_port}"
        self.timeout = timeout
        self.retry_count = retry_count

        # Shared directories (SMB/CIFS mount points)
        if shared_input_dir is None:
            shared_input_dir = Path("/mnt/forensics_vm/input")
        if shared_output_dir is None:
            shared_output_dir = Path("/mnt/forensics_vm/output")

        self.shared_input = Path(shared_input_dir)
        self.shared_output = Path(shared_output_dir)

        # Ensure shared directories exist
        self.shared_input.mkdir(parents=True, exist_ok=True)
        self.shared_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ XenVMExecutor initialized: {vm_ip}:{vm_port}")

    def check_health(self) -> bool:
        """
        Check if VM RPC server is healthy and responding

        Returns:
            True if VM is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Forensics VM healthy: {data.get('timestamp')}")
                return True
            else:
                logger.warning(f"VM health check failed: HTTP {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.warning(f"❌ Cannot connect to forensics VM at {self.vm_ip}:{self.vm_port}")
            return False
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️  Forensics VM health check timeout")
            return False
        except Exception as e:
            logger.error(f"VM health check error: {e}")
            return False

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List available tools on VM

        Returns:
            Dict of tool names to status info
        """
        try:
            response = requests.get(
                f"{self.base_url}/tools",
                timeout=10
            )

            if response.status_code == 200:
                tools = response.json()
                logger.info(f"✓ Found {len(tools)} tools on VM")
                return tools
            else:
                logger.error(f"Failed to list tools: HTTP {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {}

    def execute_tool(
        self,
        tool_name: str,
        input_file: Path,
        args: List[str] = None,
        save_output: bool = True
    ) -> VMExecutionResult:
        """
        Execute forensic tool on VM

        Workflow:
        1. Copy input file to shared folder
        2. Map to Windows path
        3. Send RPC request to VM
        4. VM executes tool
        5. Return results

        Args:
            tool_name: Name of tool (e.g., 'dbxELA', 'dbxNoiseMap')
            input_file: Path to input file on Dom0
            args: Tool command-line arguments
            save_output: Save output to shared folder

        Returns:
            VMExecutionResult with execution details
        """
        start_time = time.time()

        if args is None:
            args = []

        # Validate input file exists
        input_file = Path(input_file)
        if not input_file.exists():
            return VMExecutionResult(
                success=False,
                tool_name=tool_name,
                stdout="",
                stderr=f"Input file not found: {input_file}",
                returncode=-1,
                execution_time=0,
                vm_ip=self.vm_ip,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        # Copy input file to shared folder
        try:
            shared_input_path = self.shared_input / input_file.name
            shutil.copy(input_file, shared_input_path)
            logger.info(f"✓ Copied {input_file.name} to shared folder")
        except Exception as e:
            return VMExecutionResult(
                success=False,
                tool_name=tool_name,
                stdout="",
                stderr=f"Failed to copy input file: {e}",
                returncode=-1,
                execution_time=0,
                vm_ip=self.vm_ip,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        # Map to Windows paths
        win_input_path = f"C:\\Forensics\\input\\{input_file.name}"
        win_output_path = None

        if save_output:
            output_filename = f"result_{input_file.stem}_{int(time.time())}.json"
            win_output_path = f"C:\\Forensics\\output\\{output_filename}"

        # Build RPC request
        request_data = {
            "tool": tool_name,
            "input_file": win_input_path,
            "args": args,
            "output_file": win_output_path
        }

        # Execute with retry logic
        for attempt in range(self.retry_count):
            try:
                logger.info(f"Executing {tool_name} on VM (attempt {attempt + 1}/{self.retry_count})...")

                response = requests.post(
                    f"{self.base_url}/analyze",
                    json=request_data,
                    timeout=self.timeout
                )

                execution_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()

                    logger.info(f"✓ {tool_name} completed in {execution_time:.1f}s")

                    return VMExecutionResult(
                        success=result.get('success', False),
                        tool_name=tool_name,
                        stdout=result.get('stdout', ''),
                        stderr=result.get('stderr', ''),
                        returncode=result.get('returncode', -1),
                        execution_time=execution_time,
                        vm_ip=self.vm_ip,
                        timestamp=result.get('timestamp', '')
                    )

                else:
                    error_msg = response.text
                    logger.warning(f"VM returned HTTP {response.status_code}: {error_msg}")

                    if attempt < self.retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return VMExecutionResult(
                            success=False,
                            tool_name=tool_name,
                            stdout="",
                            stderr=f"HTTP {response.status_code}: {error_msg}",
                            returncode=-1,
                            execution_time=execution_time,
                            vm_ip=self.vm_ip,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )

            except requests.exceptions.Timeout:
                logger.error(f"⏱️  Tool execution timeout ({self.timeout}s)")

                if attempt < self.retry_count - 1:
                    continue
                else:
                    return VMExecutionResult(
                        success=False,
                        tool_name=tool_name,
                        stdout="",
                        stderr=f"Execution timeout after {self.timeout}s",
                        returncode=-1,
                        execution_time=time.time() - start_time,
                        vm_ip=self.vm_ip,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )

            except Exception as e:
                logger.error(f"VM execution error: {e}")

                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return VMExecutionResult(
                        success=False,
                        tool_name=tool_name,
                        stdout="",
                        stderr=str(e),
                        returncode=-1,
                        execution_time=time.time() - start_time,
                        vm_ip=self.vm_ip,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )

        # Should never reach here
        return VMExecutionResult(
            success=False,
            tool_name=tool_name,
            stdout="",
            stderr="Unknown error",
            returncode=-1,
            execution_time=time.time() - start_time,
            vm_ip=self.vm_ip,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def get_output_file(self, filename: str) -> Optional[Path]:
        """
        Get output file from shared folder

        Args:
            filename: Name of output file

        Returns:
            Path to output file or None if not found
        """
        output_path = self.shared_output / filename

        if output_path.exists():
            return output_path
        else:
            logger.warning(f"Output file not found: {filename}")
            return None

    def cleanup_shared_folders(self):
        """Clean up temporary files in shared folders"""
        try:
            # Clean input folder
            for file in self.shared_input.glob("*"):
                if file.is_file():
                    file.unlink()

            # Clean output folder (keep last 100 files)
            output_files = sorted(
                self.shared_output.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            for file in output_files[100:]:
                file.unlink()

            logger.info("✓ Cleaned up shared folders")

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# Convenience function for quick execution
def execute_forensic_tool(
    tool_name: str,
    input_file: Path,
    args: List[str] = None,
    vm_ip: str = "192.168.100.10"
) -> Dict[str, Any]:
    """
    Quick helper to execute forensic tool on VM

    Args:
        tool_name: Tool name (e.g., 'dbxELA')
        input_file: Input file path
        args: Tool arguments
        vm_ip: VM IP address

    Returns:
        Dict with execution results
    """
    executor = XenVMExecutor(vm_ip=vm_ip)

    if not executor.check_health():
        return {
            "success": False,
            "error": "Forensics VM not available"
        }

    result = executor.execute_tool(
        tool_name=tool_name,
        input_file=input_file,
        args=args or []
    )

    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": result.execution_time,
        "timestamp": result.timestamp
    }


if __name__ == "__main__":
    import sys

    print("=== Xen VM Executor Test ===\n")

    # Initialize executor
    executor = XenVMExecutor(vm_ip="192.168.100.10")

    # Check VM health
    print("1. Checking VM health...")
    if executor.check_health():
        print("   ✓ Forensics VM is healthy\n")
    else:
        print("   ❌ Forensics VM not available")
        print("\n   Make sure:")
        print("   1. Windows Server Core VM is running")
        print("   2. forensics_rpc_server.py is running on VM")
        print("   3. VM is accessible at 192.168.100.10:5000")
        print("   4. Shared folders are mounted at /mnt/forensics_vm/")
        sys.exit(1)

    # List available tools
    print("2. Listing available tools...")
    tools = executor.list_tools()

    for tool_name, tool_info in tools.items():
        status = "✓" if tool_info.get('exists') else "❌"
        print(f"   {status} {tool_name}: {tool_info.get('path')}")

    print("\n✓ VM executor ready for forensic analysis")
    print("\nUsage example:")
    print("  from xen_vm_executor import XenVMExecutor")
    print("  executor = XenVMExecutor()")
    print("  result = executor.execute_tool('dbxELA', Path('screenshot.jpg'))")
    print("  print(result.stdout)")
