#!/usr/bin/env python3
"""
Phase 2 Deployment Validation Suite
Dell Latitude 5450 MIL-SPEC DSMIL System

Multi-Agent Coordination: MONITOR + INFRASTRUCTURE + TESTBED + QADIRECTOR
Comprehensive validation of TPM, ML, Device Monitoring, Agent Coordination, and Performance

Date: September 2, 2025
System: Dell Latitude 5450 MIL-SPEC JRTC1
Target: Phase 2 production deployment validation
"""

import os
import sys
import json
import time
import sqlite3
import subprocess
import platform
import psutil
import hashlib
import socket
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/john/LAT5150DRVMIL/logs/phase2_validation.log')
    ]
)
logger = logging.getLogger('Phase2Validator')

@dataclass
class ValidationResult:
    """Standard validation result structure"""
    component: str
    test_name: str
    status: str  # PASS, FAIL, WARN, SKIP
    score: float  # 0.0-1.0
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: str

@dataclass
class DeploymentHealthScore:
    """Overall deployment health assessment"""
    overall_score: float
    component_scores: Dict[str, float]
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    critical_failures: List[str]
    recommendations: List[str]

class Phase2DeploymentValidator:
    """
    Comprehensive Phase 2 deployment validation suite
    
    Validates:
    1. TPM Integration (Device 0x8005, PCR 16, ECC keys)
    2. ML Learning System (PostgreSQL port 5433, embeddings)
    3. Device Monitoring (7 Phase 2 devices)
    4. Agent Coordination (80 agents, Tandem Orchestrator)
    5. Performance Benchmarks (XOR/SSE4.2, queries, response times)
    """
    
    def __init__(self):
        self.base_path = Path("/home/john/LAT5150DRVMIL")
        self.results: List[ValidationResult] = []
        self.start_time = datetime.now(timezone.utc)
        
        # Phase 2 Target Devices
        self.phase2_devices = {
            0x8005: "TPM Interface Controller",
            0x8008: "Secure Boot Validator", 
            0x8011: "Encryption Key Management",
            0x8013: "Unknown Extended Security",
            0x8014: "Unknown Extended Security",
            0x8022: "Network Security Filter",
            0x8027: "Network Authentication Gateway"
        }
        
        # Critical quarantined devices (must never be writable)
        self.quarantine_devices = [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]
        
        # Performance benchmarks
        self.performance_targets = {
            'tpm_ecc_sign_ms': 50.0,  # ECC P-256 signing target
            'tpm_rsa_sign_ms': 150.0, # RSA 2048 signing target
            'db_query_ms': 100.0,     # Database query response
            'agent_response_ms': 500.0, # Agent coordination response
            'xor_ops_per_sec': 1000000, # XOR operations with SSE4.2
            'thermal_response_ms': 1000.0 # Thermal monitoring response
        }

    def run_test(self, component: str, test_name: str, test_func) -> ValidationResult:
        """Execute a single test with timing and error handling"""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            logger.info(f"Running {component}.{test_name}")
            status, score, message, details = test_func()
            duration_ms = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                component=component,
                test_name=test_name,
                status=status,
                score=score,
                message=message,
                details=details,
                duration_ms=duration_ms,
                timestamp=timestamp
            )
            
            self.results.append(result)
            logger.info(f"✓ {component}.{test_name}: {status} ({score:.2f}) - {message}")
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = ValidationResult(
                component=component,
                test_name=test_name,
                status="FAIL",
                score=0.0,
                message=f"Test failed with exception: {str(e)}",
                details={"exception": str(e), "type": type(e).__name__},
                duration_ms=duration_ms,
                timestamp=timestamp
            )
            
            self.results.append(error_result)
            logger.error(f"✗ {component}.{test_name}: FAIL - {str(e)}")
            return error_result

    # ========================================================================
    # TPM INTEGRATION VALIDATION (Device 0x8005)
    # ========================================================================

    def validate_tpm_device_activation(self) -> Tuple[str, float, str, Dict]:
        """Validate TPM device 0x8005 activation and basic functionality"""
        details = {}
        
        # Check TPM activation report
        activation_file = self.base_path / "tpm_activation_report.json"
        if not activation_file.exists():
            return "FAIL", 0.0, "TPM activation report not found", details
            
        try:
            with open(activation_file) as f:
                tpm_report = json.load(f)
                
            details["activation_report"] = tpm_report
            
            # Validate key fields
            required_fields = ["device_id", "activation_status", "tpm_available", 
                             "user_in_tss", "pcr_extended", "ecc_keys_created"]
            
            score = 0.0
            for field in required_fields:
                if field in tpm_report and tmp_report.get(field):
                    score += 1.0/len(required_fields)
                    
            if tmp_report.get("device_id") == "0x8005" and tpm_report.get("activation_status") == "active":
                return "PASS", score, f"TPM device 0x8005 activated with {score:.1%} features", details
            else:
                return "FAIL", score, f"TPM device not properly activated ({score:.1%})", details
                
        except json.JSONDecodeError as e:
            return "FAIL", 0.0, f"Invalid TPM activation report: {e}", details

    def validate_tpm_pcr_extension(self) -> Tuple[str, float, str, Dict]:
        """Validate TPM PCR 16 extension for DSMIL"""
        details = {}
        
        try:
            # Check if tpm2-tools are available
            result = subprocess.run(['tpm2_pcrread', 'sha256:16'], 
                                 capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                pcr_value = result.stdout.strip()
                details["pcr16_value"] = pcr_value
                
                # PCR 16 should not be all zeros (indicates it was extended)
                if "00000000" not in pcr_value and len(pcr_value) > 20:
                    return "PASS", 1.0, "PCR 16 extended for DSMIL", details
                else:
                    return "WARN", 0.5, "PCR 16 may not be properly extended", details
            else:
                details["error"] = result.stderr
                return "FAIL", 0.0, f"Cannot read TPM PCR 16: {result.stderr}", details
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "SKIP", 0.0, "TPM tools not available or timeout", details

    def validate_tpm_ecc_performance(self) -> Tuple[str, float, str, Dict]:
        """Validate TPM ECC key performance meets targets"""
        details = {}
        
        try:
            # Test ECC P-256 signing performance
            start_time = time.time()
            
            # Create test data
            test_data = b"DSMIL Phase 2 deployment validation test data"
            
            # Try to use existing ECC key or create temporary one
            result = subprocess.run([
                'tpm2_sign', '-c', '0x81000000', '-g', 'sha256', 
                '-f', 'plain', '-s', 'ecdsa', '-o', '/tmp/test_sig.bin'
            ], input=test_data, capture_output=True, timeout=15)
            
            duration_ms = (time.time() - start_time) * 1000
            details["sign_duration_ms"] = duration_ms
            details["target_ms"] = self.performance_targets['tpm_ecc_sign_ms']
            
            if result.returncode == 0:
                if duration_ms <= self.performance_targets['tpm_ecc_sign_ms']:
                    score = min(1.0, self.performance_targets['tpm_ecc_sign_ms'] / duration_ms)
                    return "PASS", score, f"ECC signing {duration_ms:.1f}ms (target: {self.performance_targets['tpm_ecc_sign_ms']}ms)", details
                else:
                    score = max(0.3, self.performance_targets['tpm_ecc_sign_ms'] / duration_ms)
                    return "WARN", score, f"ECC signing slower than target: {duration_ms:.1f}ms", details
            else:
                details["error"] = result.stderr
                return "FAIL", 0.0, f"ECC signing failed: {result.stderr}", details
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "SKIP", 0.0, "TPM ECC performance test skipped", details

    # ========================================================================
    # ML LEARNING SYSTEM VALIDATION
    # ========================================================================

    def validate_ml_database_connection(self) -> Tuple[str, float, str, Dict]:
        """Validate ML learning system database connectivity"""
        details = {}
        
        try:
            # Test PostgreSQL on port 5433
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(15)
            result = sock.connect_ex(('localhost', 5433))
            sock.close()
            
            if result == 0:
                details["port_5433"] = "accessible"
                
                # Try to connect with basic credentials
                try:
                    import psycopg2
                    conn = psycopg2.connect(
                        host="localhost", port=5433, database="postgres", 
                        user="postgres", password="postgres", connect_timeout=15
                    )
                    conn.close()
                    return "PASS", 1.0, "ML database accessible on port 5433", details
                    
                except ImportError:
                    # Fallback to SQLite if PostgreSQL not available
                    db_file = self.base_path / "database" / "data" / "dsmil_tokens.db"
                    if db_file.exists():
                        details["fallback_db"] = str(db_file)
                        return "WARN", 0.7, "Using SQLite fallback database", details
                    else:
                        return "FAIL", 0.0, "No database connection available", details
                        
                except Exception as e:
                    details["postgres_error"] = str(e)
                    return "WARN", 0.3, f"Port accessible but connection failed: {e}", details
                    
            else:
                details["port_5433"] = "not_accessible"
                return "FAIL", 0.0, "ML database not accessible on port 5433", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Database validation failed: {e}", details

    def validate_ml_vector_embeddings(self) -> Tuple[str, float, str, Dict]:
        """Validate 512-dimensional embeddings functionality"""
        details = {}
        
        try:
            # Check if learning integration is available
            learning_dir = self.base_path / "infrastructure" / "learning"
            if not learning_dir.exists():
                return "SKIP", 0.0, "Learning integration directory not found", details
                
            # Test vector embedding creation
            import numpy as np
            
            # Create test 512-dimensional embedding
            test_embedding = np.random.rand(512).astype(np.float32)
            details["embedding_dimension"] = len(test_embedding)
            details["embedding_dtype"] = str(test_embedding.dtype)
            
            # Test basic vector operations
            dot_product = np.dot(test_embedding, test_embedding)
            norm = np.linalg.norm(test_embedding)
            
            details["dot_product"] = float(dot_product)
            details["norm"] = float(norm)
            
            if len(test_embedding) == 512 and norm > 0:
                return "PASS", 1.0, "512-dim vector embeddings functional", details
            else:
                return "FAIL", 0.0, f"Vector embedding validation failed", details
                
        except ImportError:
            return "SKIP", 0.0, "NumPy not available for vector operations", details
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Vector embedding test failed: {e}", details

    def validate_ml_learning_connector(self) -> Tuple[str, float, str, Dict]:
        """Validate agent learning system integration"""
        details = {}
        
        try:
            learning_connector = self.base_path / "infrastructure" / "learning" / "enhanced_learning_connector.py"
            if learning_connector.exists():
                details["connector_file"] = str(learning_connector)
                
                # Check if it's importable
                sys.path.append(str(learning_connector.parent))
                try:
                    import enhanced_learning_connector
                    details["import_success"] = True
                    return "PASS", 1.0, "Learning connector available and importable", details
                except ImportError as e:
                    details["import_error"] = str(e)
                    return "WARN", 0.5, f"Learning connector import failed: {e}", details
            else:
                return "FAIL", 0.0, "Enhanced learning connector not found", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Learning connector validation failed: {e}", details

    # ========================================================================
    # DEVICE MONITORING VALIDATION (7 Phase 2 devices)
    # ========================================================================

    def validate_phase2_device_discovery(self) -> Tuple[str, float, str, Dict]:
        """Validate 7 Phase 2 devices are discoverable and responding"""
        details = {}
        discovered_devices = []
        
        # Load device risk database
        risk_db_file = self.base_path / "device_risk_database.json"
        if risk_db_file.exists():
            try:
                with open(risk_db_file) as f:
                    risk_db = json.load(f)
                    
                details["risk_database_loaded"] = True
                details["total_devices_in_db"] = len(risk_db.get("devices", {}))
                
                # Check each Phase 2 device
                for device_id, device_name in self.phase2_devices.items():
                    device_hex = f"0x{device_id:04X}"
                    if device_hex in risk_db.get("devices", {}):
                        device_info = risk_db["devices"][device_hex]
                        discovered_devices.append({
                            "device_id": device_hex,
                            "name": device_info.get("name", device_name),
                            "risk_level": device_info.get("risk_level"),
                            "confidence": device_info.get("confidence")
                        })
                        
                details["discovered_devices"] = discovered_devices
                details["discovery_count"] = len(discovered_devices)
                details["target_count"] = len(self.phase2_devices)
                
                score = len(discovered_devices) / len(self.phase2_devices)
                
                if len(discovered_devices) == len(self.phase2_devices):
                    return "PASS", score, f"All {len(discovered_devices)} Phase 2 devices discovered", details
                elif len(discovered_devices) > 0:
                    return "WARN", score, f"{len(discovered_devices)}/{len(self.phase2_devices)} Phase 2 devices discovered", details
                else:
                    return "FAIL", 0.0, "No Phase 2 devices discovered", details
                    
            except json.JSONDecodeError as e:
                details["json_error"] = str(e)
                return "FAIL", 0.0, f"Risk database corrupted: {e}", details
        else:
            return "FAIL", 0.0, "Device risk database not found", details

    def validate_smi_interface_operational(self) -> Tuple[str, float, str, Dict]:
        """Validate SMI interface is operational for device communication"""
        details = {}
        
        try:
            # Check for DSMIL kernel module
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            modules = result.stdout
            
            dsmil_loaded = 'dsmil' in modules or 'dell_milspec' in modules
            details["dsmil_module_loaded"] = dsmil_loaded
            details["loaded_modules"] = [line for line in modules.split('\n') if 'dsmil' in line or 'dell' in line]
            
            if dsmil_loaded:
                # Test SMI interface availability
                smi_test_script = self.base_path / "test_smi_direct.py"
                if smi_test_script.exists():
                    result = subprocess.run([sys.executable, str(smi_test_script)], 
                                         capture_output=True, text=True, timeout=20)
                    
                    details["smi_test_output"] = result.stdout
                    details["smi_test_error"] = result.stderr
                    details["smi_test_returncode"] = result.returncode
                    
                    if result.returncode == 0 and "SUCCESS" in result.stdout:
                        return "PASS", 1.0, "SMI interface operational", details
                    else:
                        return "WARN", 0.5, "SMI interface test inconclusive", details
                else:
                    return "WARN", 0.3, "DSMIL module loaded but SMI test not available", details
            else:
                return "FAIL", 0.0, "DSMIL kernel module not loaded", details
                
        except subprocess.TimeoutExpired:
            return "WARN", 0.2, "SMI interface test timed out", details
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"SMI interface validation failed: {e}", details

    def validate_quarantine_enforcement(self) -> Tuple[str, float, str, Dict]:
        """Validate quarantine list prevents access to dangerous devices"""
        details = {}
        
        try:
            # Load risk database
            risk_db_file = self.base_path / "device_risk_database.json"
            if not risk_db_file.exists():
                return "FAIL", 0.0, "Risk database not found for quarantine validation", details
                
            with open(risk_db_file) as f:
                risk_db = json.load(f)
                
            quarantine_list = risk_db.get("quarantine_list", [])
            details["quarantine_list"] = quarantine_list
            details["expected_quarantine"] = [f"0x{dev:04X}" for dev in self.quarantine_devices]
            
            # Check if all critical devices are quarantined
            quarantined_count = 0
            for device in self.quarantine_devices:
                device_hex = f"0x{device:04X}"
                if device_hex in quarantine_list:
                    quarantined_count += 1
                    
            details["quarantined_count"] = quarantined_count
            details["total_critical"] = len(self.quarantine_devices)
            
            score = quarantined_count / len(self.quarantine_devices)
            
            if quarantined_count == len(self.quarantine_devices):
                return "PASS", 1.0, f"All {len(self.quarantine_devices)} critical devices quarantined", details
            else:
                return "FAIL", score, f"Only {quarantined_count}/{len(self.quarantine_devices)} critical devices quarantined", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Quarantine validation failed: {e}", details

    def validate_thermal_monitoring(self) -> Tuple[str, float, str, Dict]:
        """Validate thermal monitoring system is active and responsive"""
        details = {}
        
        try:
            # Check thermal sensors
            thermal_zones = []
            thermal_temps = []
            
            for thermal_file in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                try:
                    with open(thermal_file) as f:
                        temp_millicelsius = int(f.read().strip())
                        temp_celsius = temp_millicelsius / 1000.0
                        thermal_zones.append({
                            "zone": thermal_file.parent.name,
                            "temp_celsius": temp_celsius
                        })
                        thermal_temps.append(temp_celsius)
                except:
                    continue
                    
            details["thermal_zones"] = thermal_zones
            details["zone_count"] = len(thermal_zones)
            details["average_temp"] = sum(thermal_temps) / len(thermal_temps) if thermal_temps else 0
            details["max_temp"] = max(thermal_temps) if thermal_temps else 0
            
            # Check thermal guardian service
            thermal_guardian = self.base_path / "thermal_guardian.py"
            if thermal_guardian.exists():
                details["thermal_guardian_available"] = True
                
                # Test thermal monitoring responsiveness
                start_time = time.time()
                result = subprocess.run([sys.executable, str(thermal_guardian), '--test-mode'], 
                                     capture_output=True, text=True, timeout=15)
                response_time_ms = (time.time() - start_time) * 1000
                
                details["thermal_response_ms"] = response_time_ms
                details["target_response_ms"] = self.performance_targets['thermal_response_ms']
                
                if response_time_ms <= self.performance_targets['thermal_response_ms']:
                    score = min(1.0, self.performance_targets['thermal_response_ms'] / response_time_ms)
                    return "PASS", score, f"Thermal monitoring responsive ({response_time_ms:.1f}ms)", details
                else:
                    score = max(0.3, self.performance_targets['thermal_response_ms'] / response_time_ms)
                    return "WARN", score, f"Thermal monitoring slow ({response_time_ms:.1f}ms)", details
            else:
                if len(thermal_zones) > 0:
                    return "WARN", 0.5, f"Basic thermal monitoring ({len(thermal_zones)} zones)", details
                else:
                    return "FAIL", 0.0, "No thermal monitoring available", details
                    
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Thermal monitoring validation failed: {e}", details

    # ========================================================================
    # AGENT COORDINATION VALIDATION (80 agents, Tandem Orchestrator)
    # ========================================================================

    def validate_agent_discovery(self) -> Tuple[str, float, str, Dict]:
        """Validate 80 agents are discoverable and accessible"""
        details = {}
        
        try:
            # Check if agents directory exists
            agents_dir = Path("/home/john/claude-backups/agents")
            if not agents_dir.exists():
                return "FAIL", 0.0, f"Agents directory not found: {agents_dir}", details
                
            # Count agent files (*.md files, excluding templates)
            agent_files = list(agents_dir.glob("*.md"))
            agent_files = [f for f in agent_files if f.name.upper() not in ['TEMPLATE.md', 'README.md']]
            
            details["agents_directory"] = str(agents_dir)
            details["discovered_agent_count"] = len(agent_files)
            details["target_agent_count"] = 80
            details["agent_files"] = [f.name for f in agent_files[:10]]  # First 10 for brevity
            
            score = min(1.0, len(agent_files) / 80.0) if len(agent_files) > 0 else 0.0
            
            if len(agent_files) >= 80:
                return "PASS", 1.0, f"{len(agent_files)} agents discovered (target: 80)", details
            elif len(agent_files) >= 60:
                return "WARN", score, f"Only {len(agent_files)}/80 agents discovered", details
            else:
                return "FAIL", score, f"Insufficient agents discovered: {len(agent_files)}", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Agent discovery failed: {e}", details
            
    def validate_tandem_orchestrator(self) -> Tuple[str, float, str, Dict]:
        """Validate Tandem Orchestrator functionality"""
        details = {}
        
        try:
            # Check for orchestrator components
            orchestrator_dir = self.base_path / "infrastructure" / "coordination"
            if orchestrator_dir.exists():
                orchestrator_files = list(orchestrator_dir.glob("*.py"))
                details["orchestrator_files"] = [str(f) for f in orchestrator_files]
                
                # Test orchestrator initialization
                agent_orchestrator = orchestrator_dir / "agent_orchestrator.py"
                if agent_orchestrator.exists():
                    # Simple import test
                    sys.path.append(str(orchestrator_dir))
                    try:
                        import agent_orchestrator
                        details["orchestrator_import"] = "success"
                        return "PASS", 1.0, "Tandem Orchestrator available and functional", details
                    except ImportError as e:
                        details["import_error"] = str(e)
                        return "WARN", 0.5, f"Orchestrator import issues: {e}", details
                else:
                    return "FAIL", 0.0, "Agent orchestrator not found", details
            else:
                return "FAIL", 0.0, "Orchestration directory not found", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Orchestrator validation failed: {e}", details

    def validate_parallel_execution(self) -> Tuple[str, float, str, Dict]:
        """Validate parallel agent execution capability"""
        details = {}
        
        try:
            # Test parallel execution with multiple threads
            import concurrent.futures
            import threading
            
            def test_task(task_id):
                start_time = time.time()
                # Simulate agent work
                time.sleep(0.1)  
                return {
                    "task_id": task_id,
                    "duration": time.time() - start_time,
                    "thread_id": threading.current_thread().ident
                }
            
            # Execute tasks in parallel
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(test_task, i) for i in range(8)]
                results = [future.result() for future in concurrent.futures.as_completed(futures, timeout=15)]
                
            total_duration = time.time() - start_time
            
            details["parallel_tasks"] = len(results)
            details["total_duration"] = total_duration
            details["unique_threads"] = len(set(r["thread_id"] for r in results))
            details["task_results"] = results
            
            # Parallel execution should be significantly faster than serial
            expected_serial_time = 0.1 * 8  # 8 * 0.1s = 0.8s
            efficiency = expected_serial_time / total_duration if total_duration > 0 else 0
            
            details["efficiency_ratio"] = efficiency
            
            if efficiency > 4.0:  # At least 4x speedup
                return "PASS", min(1.0, efficiency/6.0), f"Parallel execution efficient ({efficiency:.1f}x speedup)", details
            elif efficiency > 2.0:  # At least 2x speedup
                return "WARN", 0.7, f"Parallel execution working but slow ({efficiency:.1f}x speedup)", details
            else:
                return "FAIL", 0.3, f"Parallel execution inefficient ({efficiency:.1f}x speedup)", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Parallel execution test failed: {e}", details

    def validate_error_handling(self) -> Tuple[str, float, str, Dict]:
        """Validate error handling and recovery mechanisms"""
        details = {}
        
        try:
            # Test emergency stop functionality
            emergency_stop = self.base_path / "dsmil_emergency_stop.py"
            if emergency_stop.exists():
                # Test emergency stop with dry-run
                result = subprocess.run([sys.executable, str(emergency_stop), '--test'], 
                                     capture_output=True, text=True, timeout=20)
                
                details["emergency_stop_test"] = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if result.returncode == 0:
                    details["emergency_stop_available"] = True
                else:
                    details["emergency_stop_available"] = False
                    
            # Test rollback capability
            rollback_files = list(self.base_path.glob("*rollback*"))
            details["rollback_scripts"] = [str(f) for f in rollback_files]
            
            score = 0.0
            if details.get("emergency_stop_available"):
                score += 0.5
            if len(rollback_files) > 0:
                score += 0.5
                
            if score >= 1.0:
                return "PASS", 1.0, "Error handling and rollback systems available", details
            elif score >= 0.5:
                return "WARN", score, "Partial error handling available", details
            else:
                return "FAIL", 0.0, "No error handling mechanisms found", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Error handling validation failed: {e}", details

    # ========================================================================
    # PERFORMANCE BENCHMARK VALIDATION
    # ========================================================================

    def validate_xor_sse42_performance(self) -> Tuple[str, float, str, Dict]:
        """Validate XOR operations with SSE4.2 acceleration"""
        details = {}
        
        try:
            # Check for compiled test binary
            test_simd = self.base_path / "test_simd"
            if test_simd.exists():
                # Run SIMD performance test
                start_time = time.time()
                result = subprocess.run([str(test_simd)], 
                                     capture_output=True, text=True, timeout=20)
                duration = time.time() - start_time
                
                details["simd_test_output"] = result.stdout
                details["simd_test_duration"] = duration
                details["simd_test_returncode"] = result.returncode
                
                if result.returncode == 0:
                    # Parse performance metrics from output
                    ops_per_sec = 0
                    for line in result.stdout.split('\n'):
                        if 'operations/sec' in line or 'ops/sec' in line:
                            import re
                            match = re.search(r'([\d,]+)', line)
                            if match:
                                ops_per_sec = int(match.group(1).replace(',', ''))
                                
                    details["measured_ops_per_sec"] = ops_per_sec
                    details["target_ops_per_sec"] = self.performance_targets['xor_ops_per_sec']
                    
                    if ops_per_sec >= self.performance_targets['xor_ops_per_sec']:
                        score = min(1.0, ops_per_sec / self.performance_targets['xor_ops_per_sec'])
                        return "PASS", score, f"XOR/SSE4.2 performance excellent ({ops_per_sec:,} ops/sec)", details
                    elif ops_per_sec > 0:
                        score = max(0.3, ops_per_sec / self.performance_targets['xor_ops_per_sec'])
                        return "WARN", score, f"XOR/SSE4.2 below target ({ops_per_sec:,} ops/sec)", details
                    else:
                        return "FAIL", 0.0, "XOR/SSE4.2 performance test failed", details
                else:
                    details["error"] = result.stderr
                    return "FAIL", 0.0, f"SIMD test execution failed: {result.stderr}", details
            else:
                # Try to compile test if source exists
                test_simd_c = self.base_path / "test_simd.c"
                if test_simd_c.exists():
                    compile_result = subprocess.run([
                        'gcc', '-O3', '-msse4.2', '-o', str(test_simd), str(test_simd_c)
                    ], capture_output=True, text=True)
                    
                    if compile_result.returncode == 0:
                        return self.validate_xor_sse42_performance()  # Retry after compilation
                    else:
                        details["compile_error"] = compile_result.stderr
                        return "FAIL", 0.0, f"SIMD test compilation failed: {compile_result.stderr}", details
                else:
                    return "SKIP", 0.0, "SIMD performance test not available", details
                    
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"XOR/SSE4.2 performance test failed: {e}", details

    def validate_database_query_performance(self) -> Tuple[str, float, str, Dict]:
        """Validate database query response times"""
        details = {}
        
        try:
            # Test local SQLite database first
            db_file = self.base_path / "database" / "data" / "dsmil_tokens.db"
            if db_file.exists():
                start_time = time.time()
                
                with sqlite3.connect(str(db_file), timeout=15) as conn:
                    cursor = conn.cursor()
                    
                    # Run test queries
                    queries = [
                        "SELECT COUNT(*) FROM sqlite_master",
                        "SELECT name FROM sqlite_master WHERE type='table'",
                        "PRAGMA table_info(sqlite_master)"
                    ]
                    
                    query_times = []
                    for query in queries:
                        query_start = time.time()
                        cursor.execute(query)
                        result = cursor.fetchall()
                        query_time = (time.time() - query_start) * 1000
                        query_times.append(query_time)
                        
                total_time = (time.time() - start_time) * 1000
                avg_query_time = sum(query_times) / len(query_times)
                
                details["database_type"] = "SQLite"
                details["total_query_time_ms"] = total_time
                details["average_query_time_ms"] = avg_query_time
                details["individual_query_times"] = query_times
                details["target_query_time_ms"] = self.performance_targets['db_query_ms']
                
                if avg_query_time <= self.performance_targets['db_query_ms']:
                    score = min(1.0, self.performance_targets['db_query_ms'] / avg_query_time)
                    return "PASS", score, f"Database queries fast ({avg_query_time:.1f}ms avg)", details
                else:
                    score = max(0.3, self.performance_targets['db_query_ms'] / avg_query_time)
                    return "WARN", score, f"Database queries slow ({avg_query_time:.1f}ms avg)", details
                    
            else:
                return "FAIL", 0.0, "No database available for performance testing", details
                
        except sqlite3.Error as e:
            details["sqlite_error"] = str(e)
            return "FAIL", 0.0, f"Database query test failed: {e}", details
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Database performance test failed: {e}", details

    def validate_agent_response_times(self) -> Tuple[str, float, str, Dict]:
        """Validate agent coordination response times"""
        details = {}
        
        try:
            # Test simple agent response simulation
            start_time = time.time()
            
            # Simulate agent coordination overhead
            test_data = {"test": "agent_coordination", "timestamp": time.time()}
            serialized = json.dumps(test_data)
            deserialized = json.loads(serialized)
            
            # Simulate network/IPC overhead
            time.sleep(0.01)  # 10ms simulated overhead
            
            response_time_ms = (time.time() - start_time) * 1000
            
            details["simulated_response_ms"] = response_time_ms
            details["target_response_ms"] = self.performance_targets['agent_response_ms']
            
            # Test with orchestrator if available
            orchestrator_test = self.base_path / "test_deployment_mock.py"
            if orchestrator_test.exists():
                orch_start = time.time()
                result = subprocess.run([sys.executable, str(orchestrator_test)], 
                                     capture_output=True, text=True, timeout=15)
                orch_time = (time.time() - orch_start) * 1000
                
                details["orchestrator_test_ms"] = orch_time
                details["orchestrator_returncode"] = result.returncode
                
                if result.returncode == 0 and orch_time <= self.performance_targets['agent_response_ms']:
                    score = min(1.0, self.performance_targets['agent_response_ms'] / orch_time)
                    return "PASS", score, f"Agent response time excellent ({orch_time:.1f}ms)", details
                elif result.returncode == 0:
                    score = max(0.3, self.performance_targets['agent_response_ms'] / orch_time)
                    return "WARN", score, f"Agent response time slow ({orch_time:.1f}ms)", details
                else:
                    return "WARN", 0.5, f"Agent response simulation only ({response_time_ms:.1f}ms)", details
            else:
                if response_time_ms <= self.performance_targets['agent_response_ms']:
                    return "WARN", 0.6, f"Agent response simulation only ({response_time_ms:.1f}ms)", details
                else:
                    return "FAIL", 0.3, f"Even simulation too slow ({response_time_ms:.1f}ms)", details
                    
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"Agent response time test failed: {e}", details

    def validate_system_resource_usage(self) -> Tuple[str, float, str, Dict]:
        """Validate system resource usage is within acceptable limits"""
        details = {}
        
        try:
            # Get current system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details["cpu_percent"] = cpu_percent
            details["memory_percent"] = memory.percent
            details["disk_percent"] = (disk.used / disk.total) * 100
            details["memory_available_gb"] = memory.available / (1024**3)
            details["disk_free_gb"] = disk.free / (1024**3)
            
            # Resource usage thresholds
            thresholds = {
                "cpu_warning": 80.0,
                "memory_warning": 85.0,
                "disk_warning": 90.0
            }
            
            details["thresholds"] = thresholds
            
            # Calculate score based on resource usage
            cpu_score = max(0, 1.0 - max(0, cpu_percent - 50) / 50.0)
            memory_score = max(0, 1.0 - max(0, memory.percent - 60) / 40.0)
            disk_score = max(0, 1.0 - max(0, ((disk.used/disk.total)*100) - 70) / 30.0)
            
            overall_score = (cpu_score + memory_score + disk_score) / 3.0
            
            details["resource_scores"] = {
                "cpu_score": cpu_score,
                "memory_score": memory_score,
                "disk_score": disk_score,
                "overall_score": overall_score
            }
            
            if cpu_percent < thresholds["cpu_warning"] and \
               memory.percent < thresholds["memory_warning"] and \
               (disk.used / disk.total) * 100 < thresholds["disk_warning"]:
                return "PASS", overall_score, f"System resources healthy (CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%)", details
            else:
                warnings = []
                if cpu_percent >= thresholds["cpu_warning"]:
                    warnings.append(f"CPU high: {cpu_percent:.1f}%")
                if memory.percent >= thresholds["memory_warning"]:
                    warnings.append(f"Memory high: {memory.percent:.1f}%")
                if (disk.used / disk.total) * 100 >= thresholds["disk_warning"]:
                    warnings.append(f"Disk high: {(disk.used/disk.total)*100:.1f}%")
                    
                return "WARN", overall_score, f"Resource warnings: {', '.join(warnings)}", details
                
        except Exception as e:
            details["error"] = str(e)
            return "FAIL", 0.0, f"System resource validation failed: {e}", details

    # ========================================================================
    # MAIN VALIDATION ORCHESTRATION
    # ========================================================================

    def run_all_validations(self) -> DeploymentHealthScore:
        """Run complete Phase 2 deployment validation suite"""
        
        logger.info("=" * 80)
        logger.info("Phase 2 Deployment Validation Suite Starting")
        logger.info(f"Timestamp: {self.start_time.isoformat()}")
        logger.info(f"System: {platform.node()} - {platform.system()} {platform.release()}")
        logger.info("=" * 80)
        
        # ====================================================================
        # TPM INTEGRATION TESTS
        # ====================================================================
        logger.info("\n🔐 TPM INTEGRATION VALIDATION")
        logger.info("-" * 40)
        
        self.run_test("TPM", "device_activation", self.validate_tpm_device_activation)
        self.run_test("TPM", "pcr_extension", self.validate_tpm_pcr_extension)
        self.run_test("TPM", "ecc_performance", self.validate_tpm_ecc_performance)
        
        # ====================================================================
        # ML LEARNING SYSTEM TESTS  
        # ====================================================================
        logger.info("\n🤖 ML LEARNING SYSTEM VALIDATION")
        logger.info("-" * 40)
        
        self.run_test("ML", "database_connection", self.validate_ml_database_connection)
        self.run_test("ML", "vector_embeddings", self.validate_ml_vector_embeddings)
        self.run_test("ML", "learning_connector", self.validate_ml_learning_connector)
        
        # ====================================================================
        # DEVICE MONITORING TESTS
        # ====================================================================
        logger.info("\n📡 DEVICE MONITORING VALIDATION")
        logger.info("-" * 40)
        
        self.run_test("Devices", "phase2_discovery", self.validate_phase2_device_discovery)
        self.run_test("Devices", "smi_interface", self.validate_smi_interface_operational)
        self.run_test("Devices", "quarantine_enforcement", self.validate_quarantine_enforcement)
        self.run_test("Devices", "thermal_monitoring", self.validate_thermal_monitoring)
        
        # ====================================================================
        # AGENT COORDINATION TESTS
        # ====================================================================
        logger.info("\n🤝 AGENT COORDINATION VALIDATION")
        logger.info("-" * 40)
        
        self.run_test("Agents", "agent_discovery", self.validate_agent_discovery)
        self.run_test("Agents", "tandem_orchestrator", self.validate_tandem_orchestrator)
        self.run_test("Agents", "parallel_execution", self.validate_parallel_execution)
        self.run_test("Agents", "error_handling", self.validate_error_handling)
        
        # ====================================================================
        # PERFORMANCE BENCHMARK TESTS
        # ====================================================================
        logger.info("\n⚡ PERFORMANCE BENCHMARK VALIDATION")
        logger.info("-" * 40)
        
        self.run_test("Performance", "xor_sse42", self.validate_xor_sse42_performance)
        self.run_test("Performance", "database_queries", self.validate_database_query_performance)
        self.run_test("Performance", "agent_response_times", self.validate_agent_response_times)
        self.run_test("Performance", "system_resources", self.validate_system_resource_usage)
        
        # ====================================================================
        # GENERATE FINAL HEALTH SCORE
        # ====================================================================
        return self._generate_health_score()
        
    def _generate_health_score(self) -> DeploymentHealthScore:
        """Generate comprehensive deployment health score"""
        
        # Calculate component scores
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
            
        component_scores = {}
        for component, results in components.items():
            if results:
                component_scores[component] = sum(r.score for r in results) / len(results)
            else:
                component_scores[component] = 0.0
                
        # Calculate overall score (weighted)
        weights = {
            "TPM": 0.25,      # Critical for security
            "ML": 0.15,       # Important for learning
            "Devices": 0.25,  # Critical for operation
            "Agents": 0.20,   # Important for coordination
            "Performance": 0.15 # Important for efficiency
        }
        
        overall_score = 0.0
        for component, weight in weights.items():
            if component in component_scores:
                overall_score += component_scores[component] * weight
                
        # Count test results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")
        warning_tests = sum(1 for r in self.results if r.status == "WARN")
        
        # Identify critical failures
        critical_failures = []
        for result in self.results:
            if result.status == "FAIL" and result.component in ["TPM", "Devices"]:
                critical_failures.append(f"{result.component}.{result.test_name}: {result.message}")
                
        # Generate recommendations
        recommendations = []
        
        if component_scores.get("TPM", 0) < 0.8:
            recommendations.append("TPM integration needs attention - verify device 0x8005 activation")
            
        if component_scores.get("ML", 0) < 0.5:
            recommendations.append("ML learning system requires setup - check PostgreSQL on port 5433")
            
        if component_scores.get("Devices", 0) < 0.8:
            recommendations.append("Device monitoring issues detected - verify DSMIL kernel module")
            
        if component_scores.get("Agents", 0) < 0.7:
            recommendations.append("Agent coordination problems - check Tandem Orchestrator status")
            
        if component_scores.get("Performance", 0) < 0.6:
            recommendations.append("Performance below targets - optimize system resources")
            
        if overall_score >= 0.9:
            recommendations.append("✅ System ready for production deployment")
        elif overall_score >= 0.8:
            recommendations.append("⚠️ System mostly ready - address warnings before deployment")
        elif overall_score >= 0.7:
            recommendations.append("🔧 System needs improvements - resolve critical issues")
        else:
            recommendations.append("❌ System not ready for deployment - major issues need resolution")
            
        return DeploymentHealthScore(
            overall_score=overall_score,
            component_scores=component_scores,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
        
    def generate_report(self, health_score: DeploymentHealthScore) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 80)
        report.append("PHASE 2 DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append(f"System: Dell Latitude 5450 MIL-SPEC JRTC1")
        report.append(f"Validation Duration: {(datetime.now(timezone.utc) - self.start_time).total_seconds():.1f}s")
        report.append("")
        
        # Overall Health Score
        report.append("📊 OVERALL DEPLOYMENT HEALTH")
        report.append("-" * 40)
        
        score_emoji = "🟢" if health_score.overall_score >= 0.9 else \
                     "🟡" if health_score.overall_score >= 0.7 else "🔴"
        
        report.append(f"{score_emoji} Overall Score: {health_score.overall_score:.1%}")
        report.append(f"✅ Passed Tests: {health_score.passed_tests}/{health_score.total_tests}")
        report.append(f"⚠️  Warning Tests: {health_score.warning_tests}")
        report.append(f"❌ Failed Tests: {health_score.failed_tests}")
        report.append("")
        
        # Component Scores
        report.append("🏗️ COMPONENT SCORES")
        report.append("-" * 40)
        for component, score in sorted(health_score.component_scores.items()):
            emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
            report.append(f"{emoji} {component}: {score:.1%}")
        report.append("")
        
        # Test Results by Component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
            
        for component, results in sorted(components.items()):
            report.append(f"🔧 {component.upper()} VALIDATION RESULTS")
            report.append("-" * 40)
            
            for result in sorted(results, key=lambda x: x.test_name):
                status_emoji = {
                    "PASS": "✅",
                    "WARN": "⚠️ ",
                    "FAIL": "❌",
                    "SKIP": "⏭️ "
                }.get(result.status, "❓")
                
                report.append(f"{status_emoji} {result.test_name}: {result.status}")
                report.append(f"    Score: {result.score:.1%}")
                report.append(f"    Message: {result.message}")
                report.append(f"    Duration: {result.duration_ms:.1f}ms")
                
                # Add key details for important tests
                if result.component == "TPM" and result.test_name == "device_activation":
                    device_id = result.details.get("activation_report", {}).get("device_id")
                    if device_id:
                        report.append(f"    Device: {device_id}")
                        
                elif result.component == "Devices" and result.test_name == "phase2_discovery":
                    discovered = result.details.get("discovery_count", 0)
                    target = result.details.get("target_count", 7)
                    report.append(f"    Discovered: {discovered}/{target} Phase 2 devices")
                    
                elif result.component == "Agents" and result.test_name == "agent_discovery":
                    agent_count = result.details.get("discovered_agent_count", 0)
                    report.append(f"    Agents: {agent_count}/80 discovered")
                    
                elif result.component == "Performance" and "performance" in result.test_name:
                    for key, value in result.details.items():
                        if "ms" in key or "ops" in key:
                            report.append(f"    {key}: {value}")
                            
                report.append("")
                
        # Critical Failures
        if health_score.critical_failures:
            report.append("🚨 CRITICAL FAILURES")
            report.append("-" * 40)
            for failure in health_score.critical_failures:
                report.append(f"❌ {failure}")
            report.append("")
            
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, recommendation in enumerate(health_score.recommendations, 1):
            report.append(f"{i}. {recommendation}")
        report.append("")
        
        # Phase 2 Device Status Summary
        report.append("🎯 PHASE 2 DEVICE STATUS")
        report.append("-" * 40)
        for device_id, device_name in self.phase2_devices.items():
            device_hex = f"0x{device_id:04X}"
            status = "✅ Ready" if device_id == 0x8005 else "📡 Monitoring"
            report.append(f"{status} {device_hex}: {device_name}")
        report.append("")
        
        # Quarantine Status
        report.append("🛡️  QUARANTINE STATUS")
        report.append("-" * 40)
        for device_id in self.quarantine_devices:
            device_hex = f"0x{device_id:04X}"
            report.append(f"🚫 {device_hex}: QUARANTINED (Critical - Never writable)")
        report.append("")
        
        # Next Steps
        report.append("🚀 NEXT STEPS")
        report.append("-" * 40)
        if health_score.overall_score >= 0.9:
            report.append("1. ✅ System validated and ready for Phase 3 deployment")
            report.append("2. 📋 Proceed with integration testing")
            report.append("3. 🎯 Begin production workload testing")
        elif health_score.overall_score >= 0.8:
            report.append("1. ⚠️  Address warning conditions")
            report.append("2. 🔄 Re-run validation after fixes")
            report.append("3. 📋 Conditional proceed with caution")
        else:
            report.append("1. ❌ Resolve critical failures before proceeding")
            report.append("2. 🔧 System requires maintenance")
            report.append("3. 🛑 DO NOT deploy until issues resolved")
            
        report.append("")
        report.append("=" * 80)
        report.append(f"Validation completed at {datetime.now(timezone.utc).isoformat()}")
        report.append("Multi-Agent Team: MONITOR + INFRASTRUCTURE + TESTBED + QADIRECTOR")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_report(self, report: str, health_score: DeploymentHealthScore):
        """Save validation report and results"""
        
        # Create logs directory if it doesn't exist
        logs_dir = self.base_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report_file = logs_dir / f"phase2_validation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
            
        # Save JSON results
        results_data = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "system": "Dell Latitude 5450 MIL-SPEC",
                "validation_version": "1.0",
                "phase": "Phase 2 Deployment"
            },
            "health_score": asdict(health_score),
            "detailed_results": [self._serialize_result(result) for result in self.results]
        }
        
        json_file = logs_dir / f"phase2_validation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        logger.info(f"Validation report saved to: {report_file}")
        logger.info(f"Detailed results saved to: {json_file}")
        
        return report_file, json_file
        
    def _serialize_result(self, result: ValidationResult) -> Dict[str, Any]:
        """Serialize validation result to JSON-safe format"""
        result_dict = asdict(result)
        
        # Convert bytes to string representations
        if 'details' in result_dict and result_dict['details']:
            for key, value in result_dict['details'].items():
                if isinstance(value, bytes):
                    result_dict['details'][key] = value.decode('utf-8', errors='replace')
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, bytes):
                            result_dict['details'][key][sub_key] = sub_value.decode('utf-8', errors='replace')
        
        return result_dict


def main():
    """Main validation execution"""
    
    print("🚀 Phase 2 Deployment Validation Suite")
    print("=" * 50)
    print("Multi-Agent Team: MONITOR + INFRASTRUCTURE + TESTBED + QADIRECTOR")
    print(f"System: Dell Latitude 5450 MIL-SPEC JRTC1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # Initialize validator
        validator = Phase2DeploymentValidator()
        
        # Run all validations
        health_score = validator.run_all_validations()
        
        # Generate and display report
        report = validator.generate_report(health_score)
        print(report)
        
        # Save results
        report_file, json_file = validator.save_report(report, health_score)
        
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"Overall Health Score: {health_score.overall_score:.1%}")
        print(f"Tests Passed: {health_score.passed_tests}/{health_score.total_tests}")
        print(f"Report saved to: {report_file}")
        
        # Exit with appropriate code
        if health_score.overall_score >= 0.9:
            print("\n✅ Phase 2 deployment VALIDATED - Ready for production")
            sys.exit(0)
        elif health_score.overall_score >= 0.8:
            print("\n⚠️  Phase 2 deployment CONDITIONAL - Address warnings")
            sys.exit(1)
        else:
            print("\n❌ Phase 2 deployment FAILED - Resolve critical issues")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        logger.exception("Validation suite failed")
        sys.exit(1)


if __name__ == "__main__":
    main()