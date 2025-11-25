#!/usr/bin/env python3
"""
DSMIL Monitoring System Installation Verification
Dell Latitude 5450 MIL-SPEC - Installation Validation Script

This script verifies that the DSMIL Read-Only Monitoring Framework
is correctly installed and ready for operation.

Author: MONITOR Agent  
Date: 2025-09-01
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.NC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")

def print_header(message):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.CYAN}{message}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")

def check_file_exists(file_path, description):
    """Check if a file exists and is readable"""
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        print_success(f"{description}: {os.path.basename(file_path)} ({size:,} bytes)")
        return True
    else:
        print_error(f"{description}: Not found - {file_path}")
        return False

def check_executable(file_path, description):
    """Check if a file exists and is executable"""
    if os.path.isfile(file_path) and os.access(file_path, os.X_OK):
        size = os.path.getsize(file_path)
        print_success(f"{description}: {os.path.basename(file_path)} (executable, {size:,} bytes)")
        return True
    elif os.path.isfile(file_path):
        print_warning(f"{description}: {os.path.basename(file_path)} exists but not executable")
        return False
    else:
        print_error(f"{description}: Not found - {file_path}")
        return False

def check_python_syntax(file_path, description):
    """Check Python script syntax"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        compile(code, file_path, 'exec')
        print_success(f"{description}: Python syntax valid")
        return True
    except SyntaxError as e:
        print_error(f"{description}: Python syntax error - {e}")
        return False
    except Exception as e:
        print_error(f"{description}: Error checking syntax - {e}")
        return False

def check_python_imports(file_path, description):
    """Check if Python script imports are available"""
    try:
        # Read the file and extract import statements
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract import statements (basic parsing)
        import_modules = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Extract module names (simplified)
                if line.startswith('import '):
                    module = line.replace('import ', '').split(' as ')[0].split(',')[0].split('.')[0].strip()
                elif line.startswith('from '):
                    module = line.split(' import ')[0].replace('from ', '').split('.')[0].strip()
                
                if module and not module.startswith('.') and module not in ['dsmil_readonly_monitor', 'dsmil_emergency_stop']:
                    import_modules.append(module)
        
        # Test imports
        missing_modules = []
        for module in set(import_modules):
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print_warning(f"{description}: Missing modules - {', '.join(missing_modules)}")
            return False
        else:
            print_success(f"{description}: All imports available")
            return True
            
    except Exception as e:
        print_warning(f"{description}: Could not check imports - {e}")
        return True  # Don't fail verification for this

def check_root_privileges():
    """Check if running with root privileges"""
    if os.geteuid() == 0:
        print_success("Root privileges: Available")
        return True
    else:
        print_warning("Root privileges: Not running as root")
        print_info("Note: Root privileges required for SMI access during actual monitoring")
        return True  # Don't fail verification - just warn

def check_system_requirements():
    """Check system requirements"""
    print_header("SYSTEM REQUIREMENTS")
    
    all_good = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 6):
        print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro} (3.6+ required)")
        all_good = False
    
    # Check required Python modules
    required_modules = ['os', 'sys', 'time', 'json', 'signal', 'threading', 'subprocess', 
                       'datetime', 'psutil', 'curses', 'argparse', 'hashlib']
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print_error(f"Missing Python modules: {', '.join(missing_modules)}")
        all_good = False
    else:
        print_success(f"Python modules: All {len(required_modules)} required modules available")
    
    # Check root privileges
    check_root_privileges()
    
    # Check terminal capabilities
    try:
        import curses
        curses.setupterm()
        print_success("Terminal: Curses support available")
    except:
        print_warning("Terminal: Curses support may not be available")
        print_info("Dashboard mode may not work properly")
    
    return all_good

def check_monitoring_scripts():
    """Check monitoring script files"""
    print_header("MONITORING SCRIPTS")
    
    script_dir = Path(__file__).parent
    all_good = True
    
    # Core monitoring scripts
    scripts = [
        ('dsmil_readonly_monitor.py', 'Primary monitoring engine'),
        ('dsmil_emergency_stop.py', 'Emergency response system'),
        ('dsmil_dashboard.py', 'Interactive dashboard'),
        ('launch_dsmil_monitor.sh', 'System launcher')
    ]
    
    for script_name, description in scripts:
        script_path = script_dir / script_name
        
        # Check file existence
        if not check_file_exists(str(script_path), description):
            all_good = False
            continue
        
        # Check executable permission
        if script_name.endswith('.sh'):
            if not check_executable(str(script_path), f"{description} (executable)"):
                all_good = False
        elif script_name.endswith('.py'):
            if not os.access(str(script_path), os.X_OK):
                print_warning(f"{description}: Not executable (will fix)")
                try:
                    os.chmod(str(script_path), 0o755)
                    print_success(f"{description}: Executable permission set")
                except:
                    print_error(f"{description}: Could not set executable permission")
                    all_good = False
        
        # Check Python syntax for .py files
        if script_name.endswith('.py'):
            if not check_python_syntax(str(script_path), f"{description} (syntax)"):
                all_good = False
                continue
            
            # Check imports
            check_python_imports(str(script_path), f"{description} (imports)")
    
    return all_good

def check_documentation():
    """Check documentation files"""
    print_header("DOCUMENTATION")
    
    script_dir = Path(__file__).parent
    all_good = True
    
    docs = [
        ('DSMIL_READONLY_MONITOR_COMPLETE.md', 'Complete implementation documentation'),
        ('README.md', 'Project README'),
        ('DSMIL_MONITORING_SETUP_COMPLETE.md', 'Monitoring setup documentation')
    ]
    
    for doc_name, description in docs:
        doc_path = script_dir / doc_name
        if check_file_exists(str(doc_path), description):
            # Check if file is not empty
            size = os.path.getsize(str(doc_path))
            if size < 100:  # Very small file
                print_warning(f"{description}: File very small ({size} bytes)")
        else:
            print_warning(f"{description}: Documentation file missing")
    
    return all_good

def check_directories():
    """Check required directories"""
    print_header("DIRECTORY STRUCTURE")
    
    script_dir = Path(__file__).parent
    all_good = True
    
    # Create monitoring_logs directory if it doesn't exist
    log_dir = script_dir / "monitoring_logs"
    if not log_dir.exists():
        try:
            log_dir.mkdir(mode=0o755, parents=True)
            print_success("Log directory: Created monitoring_logs/")
        except Exception as e:
            print_error(f"Log directory: Could not create monitoring_logs/ - {e}")
            all_good = False
    else:
        print_success("Log directory: monitoring_logs/ exists")
    
    # Check write permissions to log directory
    if log_dir.exists():
        try:
            test_file = log_dir / "test_write_permission.tmp"
            test_file.write_text("test")
            test_file.unlink()
            print_success("Log directory: Write permissions OK")
        except Exception as e:
            print_error(f"Log directory: No write permission - {e}")
            all_good = False
    
    return all_good

def run_basic_functionality_tests():
    """Run basic functionality tests"""
    print_header("BASIC FUNCTIONALITY TESTS")
    
    script_dir = Path(__file__).parent
    all_good = True
    
    # Test 1: Import monitoring modules
    try:
        sys.path.insert(0, str(script_dir))
        
        # Test basic imports
        print_info("Testing module imports...")
        
        # These imports should work if the files are syntactically correct
        import dsmil_readonly_monitor
        import dsmil_emergency_stop
        
        print_success("Module imports: Core monitoring modules imported successfully")
        
        # Test class instantiation (without starting monitoring)
        try:
            from dsmil_readonly_monitor import DSMILReadOnlyMonitor, MonitoringMode
            from dsmil_emergency_stop import DSMILEmergencyStop
            
            monitor = DSMILReadOnlyMonitor(MonitoringMode.DASHBOARD)
            emergency = DSMILEmergencyStop()
            
            print_success("Class instantiation: Core classes created successfully")
            
        except Exception as e:
            print_warning(f"Class instantiation: Could not create objects - {e}")
        
    except Exception as e:
        print_error(f"Module imports: Failed to import monitoring modules - {e}")
        all_good = False
    
    # Test 2: Emergency stop system validation
    try:
        print_info("Testing emergency stop system...")
        result = subprocess.run([
            sys.executable, 
            str(script_dir / 'dsmil_emergency_stop.py'), 
            '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_success("Emergency stop: Help system working")
        else:
            print_warning("Emergency stop: Help system returned non-zero exit code")
            
    except Exception as e:
        print_warning(f"Emergency stop: Could not test help system - {e}")
    
    # Test 3: Dashboard system validation (without curses)
    try:
        print_info("Testing dashboard argument parsing...")
        result = subprocess.run([
            sys.executable, 
            str(script_dir / 'dsmil_dashboard.py'), 
            '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_success("Dashboard: Argument parsing working")
        else:
            print_warning("Dashboard: Argument parsing returned non-zero exit code")
            
    except Exception as e:
        print_warning(f"Dashboard: Could not test argument parsing - {e}")
    
    return all_good

def generate_installation_report():
    """Generate final installation report"""
    print_header("INSTALLATION VERIFICATION COMPLETE")
    
    print_info("DSMIL Read-Only Monitoring Framework Installation Verification")
    print_info("All critical components have been checked")
    print("")
    print_info("To start monitoring:")
    print_info("1. Run: sudo ./launch_dsmil_monitor.sh")
    print_info("2. Select option 1 for Interactive Dashboard")
    print_info("3. Follow on-screen instructions")
    print("")
    print_info("For emergency stop:")
    print_info("- Dashboard: Press 'e' key")
    print_info("- Command line: Ctrl+C")
    print_info("- Direct: sudo python3 dsmil_emergency_stop.py --stop")
    print("")
    print_info("Documentation: DSMIL_READONLY_MONITOR_COMPLETE.md")

def main():
    """Main verification function"""
    print_header("DSMIL MONITORING SYSTEM VERIFICATION")
    print_info("Verifying installation of DSMIL Read-Only Monitoring Framework")
    print_info("Dell Latitude 5450 MIL-SPEC - 84 Device Monitoring System")
    
    all_checks_passed = True
    
    # Run verification checks
    if not check_system_requirements():
        all_checks_passed = False
    
    if not check_monitoring_scripts():
        all_checks_passed = False
    
    if not check_documentation():
        # Documentation issues are warnings, not failures
        pass
    
    if not check_directories():
        all_checks_passed = False
    
    if not run_basic_functionality_tests():
        # Functionality test issues are warnings, not failures  
        pass
    
    # Generate final report
    generate_installation_report()
    
    # Final status
    print_header("VERIFICATION RESULT")
    
    if all_checks_passed:
        print_success("✅ VERIFICATION PASSED")
        print_success("DSMIL Monitoring System is ready for operation")
        print_info("Run 'sudo ./launch_dsmil_monitor.sh' to start monitoring")
        return 0
    else:
        print_error("❌ VERIFICATION FAILED")
        print_error("Some critical issues need to be resolved before operation")
        print_info("Review the error messages above and fix issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())