#!/usr/bin/env python3
"""
TPM2 Acceleration Library Deployment Example
Demonstrates production deployment and integration patterns

Author: C-INTERNAL Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.python_bindings import (
    create_accelerated_library, TPM2LibraryConfig, TPM2SecurityLevel,
    TPM2AccelerationFlags, TPM2AcceleratedSession, TPM2PCRBank
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    security_level: TPM2SecurityLevel = TPM2SecurityLevel.CONFIDENTIAL
    max_concurrent_sessions: int = 16
    enable_hardware_acceleration: bool = True
    enable_performance_monitoring: bool = True
    enable_fault_detection: bool = True
    memory_pool_size_mb: int = 64
    log_file_path: Optional[str] = "/var/log/tpm2_acceleration.log"

class ProductionTPMService:
    """Production TPM service with acceleration"""

    def __init__(self, config: DeploymentConfig):
        """Initialize production service"""
        self.config = config
        self.library = None
        self.active_sessions = {}
        self.session_lock = threading.RLock()
        self.performance_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_latency_ms": 0.0,
            "peak_latency_ms": 0.0,
            "throughput_ops_per_second": 0.0
        }
        self.metrics_lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize the service"""
        try:
            logger.info("Initializing TPM2 Acceleration Service...")

            # Create library configuration
            lib_config = TPM2LibraryConfig(
                security_level=self.config.security_level,
                acceleration_flags=TPM2AccelerationFlags.ALL if self.config.enable_hardware_acceleration else TPM2AccelerationFlags.NONE,
                enable_profiling=self.config.enable_performance_monitoring,
                enable_fault_detection=self.config.enable_fault_detection,
                max_sessions=self.config.max_concurrent_sessions,
                memory_pool_size_mb=self.config.memory_pool_size_mb,
                log_file_path=self.config.log_file_path,
                enable_debug_mode=False
            )

            # Initialize library
            self.library = create_accelerated_library(lib_config)

            logger.info(f"Service initialized successfully")
            logger.info(f"Library version: {self.library.get_version()}")
            logger.info(f"Security level: {self.config.security_level.name}")
            logger.info(f"Hardware acceleration: {'Enabled' if self.config.enable_hardware_acceleration else 'Disabled'}")

            return True

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

    def create_session(self, session_id: Optional[str] = None) -> Optional[str]:
        """Create a new TPM session"""
        try:
            with self.session_lock:
                if len(self.active_sessions) >= self.config.max_concurrent_sessions:
                    logger.warning("Maximum concurrent sessions reached")
                    return None

                # Generate session ID if not provided
                if session_id is None:
                    session_id = f"session_{int(time.time() * 1000000)}"

                # Establish ME session
                me_session_id = self.library.establish_me_session(self.config.security_level)
                if me_session_id:
                    self.active_sessions[session_id] = {
                        "me_session_id": me_session_id,
                        "created_time": time.time(),
                        "last_activity": time.time(),
                        "operation_count": 0
                    }

                    logger.info(f"Session created: {session_id}")
                    return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {e}")

        return None

    def close_session(self, session_id: str) -> bool:
        """Close a TPM session"""
        try:
            with self.session_lock:
                if session_id not in self.active_sessions:
                    logger.warning(f"Session not found: {session_id}")
                    return False

                session_info = self.active_sessions[session_id]
                me_session_id = session_info["me_session_id"]

                # Close ME session
                self.library.close_me_session(me_session_id)

                # Remove from active sessions
                del self.active_sessions[session_id]

                logger.info(f"Session closed: {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
            return False

    def translate_pcr_batch(self, pcr_decimals: List[int], bank: TPM2PCRBank = TPM2PCRBank.SHA256) -> Optional[List[int]]:
        """High-performance batch PCR translation"""
        start_time = time.perf_counter()

        try:
            # Use hardware-accelerated batch translation
            hex_pcrs = self.library.pcr_translate_batch(pcr_decimals, bank)

            # Update performance metrics
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["successful_operations"] += 1
                self._update_latency_metrics(latency_ms)

            logger.debug(f"Batch translated {len(pcr_decimals)} PCRs in {latency_ms:.2f}ms")
            return hex_pcrs

        except Exception as e:
            logger.error(f"Batch PCR translation failed: {e}")

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["failed_operations"] += 1

            return None

    def process_tpm_command(self, session_id: str, tpm_command: bytes, timeout_ms: int = 5000) -> Optional[bytes]:
        """Process TPM command through accelerated pipeline"""
        start_time = time.perf_counter()

        try:
            with self.session_lock:
                if session_id not in self.active_sessions:
                    raise ValueError(f"Invalid session ID: {session_id}")

                session_info = self.active_sessions[session_id]
                me_session_id = session_info["me_session_id"]

                # Update session activity
                session_info["last_activity"] = time.time()
                session_info["operation_count"] += 1

            # Process command through accelerated ME interface
            response = self.library.send_tpm_command_via_me(me_session_id, tpm_command, timeout_ms)

            # Update performance metrics
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["successful_operations"] += 1
                self._update_latency_metrics(latency_ms)

            logger.debug(f"TPM command processed in {latency_ms:.2f}ms")
            return response

        except Exception as e:
            logger.error(f"TPM command processing failed: {e}")

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["failed_operations"] += 1

            return None

    def compute_hash_accelerated(self, data: bytes, algorithm: str = "SHA256") -> Optional[bytes]:
        """Hardware-accelerated hash computation"""
        start_time = time.perf_counter()

        try:
            hash_result = self.library.compute_hash_accelerated(data, algorithm)

            # Update performance metrics
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["successful_operations"] += 1
                self._update_latency_metrics(latency_ms)

            logger.debug(f"{algorithm} hash computed in {latency_ms:.2f}ms")
            return hash_result

        except Exception as e:
            logger.error(f"Hash computation failed: {e}")

            with self.metrics_lock:
                self.performance_metrics["total_operations"] += 1
                self.performance_metrics["failed_operations"] += 1

            return None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        with self.metrics_lock:
            return self.performance_metrics.copy()

    def get_session_status(self) -> Dict:
        """Get session status information"""
        with self.session_lock:
            return {
                "active_sessions": len(self.active_sessions),
                "max_sessions": self.config.max_concurrent_sessions,
                "session_details": {
                    sid: {
                        "created_time": info["created_time"],
                        "last_activity": info["last_activity"],
                        "operation_count": info["operation_count"],
                        "age_seconds": time.time() - info["created_time"]
                    }
                    for sid, info in self.active_sessions.items()
                }
            }

    def cleanup_expired_sessions(self, max_age_seconds: int = 3600):
        """Cleanup expired sessions"""
        current_time = time.time()
        expired_sessions = []

        with self.session_lock:
            for session_id, session_info in self.active_sessions.items():
                if current_time - session_info["last_activity"] > max_age_seconds:
                    expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            self.close_session(session_id)

    def shutdown(self):
        """Shutdown the service"""
        logger.info("Shutting down TPM2 Acceleration Service...")

        # Close all active sessions
        with self.session_lock:
            session_ids = list(self.active_sessions.keys())

        for session_id in session_ids:
            self.close_session(session_id)

        # Cleanup library
        if self.library:
            self.library.cleanup()

        logger.info("Service shutdown complete")

    def _update_latency_metrics(self, latency_ms: float):
        """Update latency performance metrics"""
        total_ops = self.performance_metrics["total_operations"]

        # Update average latency
        current_avg = self.performance_metrics["average_latency_ms"]
        new_avg = (current_avg * (total_ops - 1) + latency_ms) / total_ops
        self.performance_metrics["average_latency_ms"] = new_avg

        # Update peak latency
        if latency_ms > self.performance_metrics["peak_latency_ms"]:
            self.performance_metrics["peak_latency_ms"] = latency_ms

        # Calculate throughput (operations per second)
        if total_ops >= 10:  # Avoid division by very small numbers
            self.performance_metrics["throughput_ops_per_second"] = 1000.0 / new_avg

def demo_basic_operations():
    """Demonstrate basic TPM operations"""
    print("=== Basic Operations Demo ===")

    config = DeploymentConfig(
        security_level=TPM2SecurityLevel.CONFIDENTIAL,
        enable_hardware_acceleration=True,
        enable_performance_monitoring=True
    )

    service = ProductionTPMService(config)

    try:
        # Initialize service
        if not service.initialize():
            print("Failed to initialize service")
            return

        # Create session
        session_id = service.create_session()
        if not session_id:
            print("Failed to create session")
            return

        print(f"Session created: {session_id}")

        # PCR translation operations
        pcr_decimals = [0, 1, 2, 3, 7, 16, 23]
        hex_pcrs = service.translate_pcr_batch(pcr_decimals)
        if hex_pcrs:
            print(f"PCR translations: {dict(zip(pcr_decimals, [f'0x{h:04X}' for h in hex_pcrs]))}")

        # Hash computation
        test_data = b"This is test data for hash computation"
        algorithms = ["SHA256", "SHA384", "SHA512"]

        for alg in algorithms:
            hash_result = service.compute_hash_accelerated(test_data, alg)
            if hash_result:
                print(f"{alg}: {hash_result.hex()[:16]}...")

        # TPM command processing
        tpm_startup_command = b'\x80\x01\x00\x00\x00\x0c\x00\x00\x01\x43\x00\x00'
        response = service.process_tpm_command(session_id, tpm_startup_command)
        if response:
            print(f"TPM response: {len(response)} bytes")

        # Display performance metrics
        metrics = service.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"  Total operations: {metrics['total_operations']}")
        print(f"  Success rate: {metrics['successful_operations'] / max(metrics['total_operations'], 1) * 100:.1f}%")
        print(f"  Average latency: {metrics['average_latency_ms']:.2f}ms")
        print(f"  Peak latency: {metrics['peak_latency_ms']:.2f}ms")

        # Close session
        service.close_session(session_id)

    finally:
        service.shutdown()

def demo_concurrent_operations():
    """Demonstrate concurrent high-performance operations"""
    print("\n=== Concurrent Operations Demo ===")

    config = DeploymentConfig(
        security_level=TPM2SecurityLevel.CONFIDENTIAL,
        max_concurrent_sessions=8,
        enable_hardware_acceleration=True
    )

    service = ProductionTPMService(config)

    try:
        if not service.initialize():
            print("Failed to initialize service")
            return

        def worker_thread(worker_id: int, operations_per_worker: int):
            """Worker thread for concurrent operations"""
            results = []

            # Create session for this worker
            session_id = service.create_session(f"worker_{worker_id}")
            if not session_id:
                return results

            try:
                for i in range(operations_per_worker):
                    # PCR translation
                    pcr = i % 24
                    hex_pcrs = service.translate_pcr_batch([pcr])

                    # Hash computation
                    data = f"worker_{worker_id}_operation_{i}".encode()
                    hash_result = service.compute_hash_accelerated(data, "SHA256")

                    if hex_pcrs and hash_result:
                        results.append({
                            "worker_id": worker_id,
                            "operation": i,
                            "pcr_hex": hex_pcrs[0],
                            "hash": hash_result.hex()[:8]
                        })

            finally:
                service.close_session(session_id)

            return results

        # Run concurrent operations
        num_workers = 4
        operations_per_worker = 50

        print(f"Running {num_workers} workers with {operations_per_worker} operations each...")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_thread, worker_id, operations_per_worker)
                for worker_id in range(num_workers)
            ]

            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)

        end_time = time.time()

        total_operations = len(all_results)
        total_time = end_time - start_time
        throughput = total_operations / total_time

        print(f"Completed {total_operations} operations in {total_time:.2f}s")
        print(f"Throughput: {throughput:.0f} operations/second")

        # Display final metrics
        metrics = service.get_performance_metrics()
        session_status = service.get_session_status()

        print(f"\nFinal Performance Metrics:")
        print(f"  Total operations: {metrics['total_operations']}")
        print(f"  Average latency: {metrics['average_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_ops_per_second']:.0f} ops/sec")

        print(f"\nSession Status:")
        print(f"  Active sessions: {session_status['active_sessions']}")

    finally:
        service.shutdown()

def demo_error_handling():
    """Demonstrate error handling and recovery"""
    print("\n=== Error Handling Demo ===")

    config = DeploymentConfig(
        security_level=TPM2SecurityLevel.UNCLASSIFIED,
        enable_fault_detection=True
    )

    service = ProductionTPMService(config)

    try:
        if not service.initialize():
            print("Failed to initialize service")
            return

        session_id = service.create_session()
        if not session_id:
            print("Failed to create session")
            return

        # Test invalid PCR values
        print("Testing invalid PCR values...")
        invalid_pcrs = [-1, 24, 100]
        for pcr in invalid_pcrs:
            try:
                result = service.translate_pcr_batch([pcr])
                print(f"  PCR {pcr}: Unexpected success")
            except Exception as e:
                print(f"  PCR {pcr}: Correctly caught error - {type(e).__name__}")

        # Test invalid hash algorithms
        print("\nTesting invalid hash algorithms...")
        invalid_algs = ["MD5", "INVALID", ""]
        test_data = b"test data"

        for alg in invalid_algs:
            try:
                result = service.compute_hash_accelerated(test_data, alg)
                print(f"  {alg}: Unexpected success")
            except Exception as e:
                print(f"  {alg}: Correctly caught error - {type(e).__name__}")

        # Test session cleanup
        print("\nTesting session cleanup...")
        service.close_session(session_id)

        # Try to use closed session
        try:
            service.process_tpm_command(session_id, b"test")
            print("  Closed session: Unexpected success")
        except Exception as e:
            print(f"  Closed session: Correctly caught error - {type(e).__name__}")

        # Display error metrics
        metrics = service.get_performance_metrics()
        if metrics['failed_operations'] > 0:
            error_rate = metrics['failed_operations'] / metrics['total_operations'] * 100
            print(f"\nError rate: {error_rate:.1f}% ({metrics['failed_operations']}/{metrics['total_operations']})")

    finally:
        service.shutdown()

def main():
    """Main deployment example"""
    print("TPM2 Acceleration Library - Production Deployment Example")
    print("=" * 60)

    try:
        # Run demonstration scenarios
        demo_basic_operations()
        demo_concurrent_operations()
        demo_error_handling()

        print("\n✓ All deployment examples completed successfully")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()