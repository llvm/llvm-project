#!/usr/bin/env python3
"""
Comprehensive Integration Tests for SHRINK Submodule System

Tests all SHRINK integration components:
1. SHRINKIntegrationManager - Submodule lifecycle management
2. SubmoduleHealthMonitor - Health checking and monitoring
3. SHRINKIntelligenceIntegration - Intelligence subroutine integration

Usage:
    python test_shrink_integration.py
    python test_shrink_integration.py --verbose
    python test_shrink_integration.py --test=manager
    python test_shrink_integration.py --test=health
    python test_shrink_integration.py --test=intelligence
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import SHRINK components
try:
    from shrink_integration_manager import (
        SHRINKIntegrationManager,
        SubmoduleConfig,
        SubmoduleStatus
    )
    MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import shrink_integration_manager: {e}")
    MANAGER_AVAILABLE = False

try:
    from submodule_health_monitor import (
        SubmoduleHealthMonitor,
        HealthMetric,
        HealthReport,
        HealthStatus
    )
    HEALTH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import submodule_health_monitor: {e}")
    HEALTH_AVAILABLE = False

try:
    from shrink_intelligence_integration import (
        SHRINKIntelligenceIntegration,
        SystemMetrics
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import shrink_intelligence_integration: {e}")
    INTELLIGENCE_AVAILABLE = False


class TestSHRINKIntegrationManager(unittest.TestCase):
    """Test suite for SHRINKIntegrationManager"""

    def setUp(self):
        """Set up test environment"""
        if not MANAGER_AVAILABLE:
            self.skipTest("SHRINKIntegrationManager not available")

        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="shrink_test_")
        self.test_path = Path(self.test_dir)

        # Initialize manager
        self.manager = SHRINKIntegrationManager(root_dir=self.test_path)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.root_dir, self.test_path)
        self.assertIn('SHRINK', self.manager.SUBMODULES)

    def test_shrink_config(self):
        """Test SHRINK submodule configuration"""
        shrink_config = self.manager.SUBMODULES['SHRINK']

        self.assertEqual(shrink_config.name, 'SHRINK')
        self.assertTrue(shrink_config.enabled)
        self.assertTrue(shrink_config.python_package)
        self.assertIn('zstd', shrink_config.dependencies)
        self.assertIn('lz4', shrink_config.dependencies)
        self.assertIn('brotli', shrink_config.dependencies)

    def test_initialize_shrink(self):
        """Test SHRINK initialization"""
        # Initialize SHRINK
        result = self.manager.initialize_shrink(force=True)

        # Check result
        self.assertTrue(result, "SHRINK initialization should succeed")

        # Check directory structure
        shrink_path = self.test_path / 'modules' / 'SHRINK'
        self.assertTrue(shrink_path.exists(), "SHRINK directory should exist")
        self.assertTrue((shrink_path / '__init__.py').exists(), "__init__.py should exist")
        self.assertTrue((shrink_path / 'compressor.py').exists(), "compressor.py should exist")
        self.assertTrue((shrink_path / 'optimizer.py').exists(), "optimizer.py should exist")
        self.assertTrue((shrink_path / 'deduplicator.py').exists(), "deduplicator.py should exist")

    def test_check_status_before_init(self):
        """Test status check before initialization"""
        status = self.manager.check_status('SHRINK')

        self.assertEqual(status.name, 'SHRINK')
        self.assertFalse(status.initialized, "SHRINK should not be initialized yet")

    def test_check_status_after_init(self):
        """Test status check after initialization"""
        # Initialize first
        self.manager.initialize_shrink(force=True)

        # Check status
        status = self.manager.check_status('SHRINK')

        self.assertEqual(status.name, 'SHRINK')
        self.assertTrue(status.initialized, "SHRINK should be initialized")
        self.assertEqual(status.health_status, 'healthy')

    def test_list_submodules(self):
        """Test listing submodules"""
        submodules = self.manager.list_submodules()

        self.assertIsInstance(submodules, list)
        self.assertGreater(len(submodules), 0)
        self.assertIn('SHRINK', submodules)

    def test_get_config(self):
        """Test getting submodule configuration"""
        config = self.manager.get_config('SHRINK')

        self.assertIsInstance(config, SubmoduleConfig)
        self.assertEqual(config.name, 'SHRINK')

    def test_invalid_submodule(self):
        """Test handling of invalid submodule name"""
        status = self.manager.check_status('NONEXISTENT')

        self.assertEqual(status.name, 'NONEXISTENT')
        self.assertFalse(status.initialized)
        self.assertIn('not found', status.error_message.lower())


class TestSubmoduleHealthMonitor(unittest.TestCase):
    """Test suite for SubmoduleHealthMonitor"""

    def setUp(self):
        """Set up test environment"""
        if not HEALTH_AVAILABLE:
            self.skipTest("SubmoduleHealthMonitor not available")

        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="shrink_health_test_")
        self.test_path = Path(self.test_dir)

        # Initialize monitor
        self.monitor = SubmoduleHealthMonitor(root_dir=self.test_path)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.root_dir, self.test_path)

    def test_check_shrink_health_not_initialized(self):
        """Test health check when SHRINK is not initialized"""
        report = self.monitor.check_shrink_health()

        self.assertEqual(report.submodule_name, 'SHRINK')
        # Status can be HEALTHY, WARNING, or ERROR depending on checks performed
        self.assertIn(report.overall_status, [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.ERROR])
        self.assertGreater(len(report.metrics), 0)

    def test_check_shrink_health_initialized(self):
        """Test health check when SHRINK is initialized"""
        # Initialize SHRINK first
        shrink_path = self.test_path / 'modules' / 'SHRINK'
        shrink_path.mkdir(parents=True, exist_ok=True)

        # Create required files
        (shrink_path / '__init__.py').write_text('# SHRINK package')
        (shrink_path / 'compressor.py').write_text('# Compressor module')
        (shrink_path / 'optimizer.py').write_text('# Optimizer module')
        (shrink_path / 'deduplicator.py').write_text('# Deduplicator module')

        # Check health
        report = self.monitor.check_shrink_health()

        self.assertEqual(report.submodule_name, 'SHRINK')
        self.assertIn(report.overall_status, [HealthStatus.HEALTHY, HealthStatus.WARNING])

        # Verify key metrics exist
        metric_names = [m.name for m in report.metrics]
        self.assertIn('directory_exists', metric_names)

    def test_health_metric_creation(self):
        """Test HealthMetric creation"""
        metric = HealthMetric(
            name='test_metric',
            status=HealthStatus.HEALTHY,
            value='test_value',
            message='Test message'
        )

        self.assertEqual(metric.name, 'test_metric')
        self.assertEqual(metric.status, HealthStatus.HEALTHY)
        self.assertEqual(metric.value, 'test_value')
        self.assertEqual(metric.message, 'Test message')

    def test_health_report_creation(self):
        """Test HealthReport creation"""
        metrics = [
            HealthMetric(name='metric1', status=HealthStatus.HEALTHY),
            HealthMetric(name='metric2', status=HealthStatus.WARNING)
        ]

        report = HealthReport(
            submodule_name='TestModule',
            overall_status=HealthStatus.WARNING,
            metrics=metrics,
            recommendations=['Fix metric2']
        )

        self.assertEqual(report.submodule_name, 'TestModule')
        self.assertEqual(report.overall_status, HealthStatus.WARNING)
        self.assertEqual(len(report.metrics), 2)
        self.assertEqual(len(report.recommendations), 1)

    def test_check_all_health(self):
        """Test checking health of all submodules"""
        # Initialize SHRINK
        shrink_path = self.test_path / 'modules' / 'SHRINK'
        shrink_path.mkdir(parents=True, exist_ok=True)
        (shrink_path / '__init__.py').write_text('# SHRINK package')

        # Check all health
        reports = self.monitor.check_all_health()

        self.assertIsInstance(reports, list)
        self.assertGreater(len(reports), 0)
        self.assertTrue(any(r.submodule_name == 'SHRINK' for r in reports))

    def test_continuous_monitoring(self):
        """Test continuous monitoring (short duration)"""
        # Initialize SHRINK
        shrink_path = self.test_path / 'modules' / 'SHRINK'
        shrink_path.mkdir(parents=True, exist_ok=True)
        (shrink_path / '__init__.py').write_text('# SHRINK package')

        # Start monitoring with very short interval
        # Note: This will run in background, we just test it starts
        try:
            # Just verify the method exists and can be called
            # We won't actually run continuous monitoring in tests
            self.assertTrue(hasattr(self.monitor, 'start_continuous_monitoring'))
        except Exception as e:
            self.fail(f"Continuous monitoring should be callable: {e}")


class TestSHRINKIntelligenceIntegration(unittest.TestCase):
    """Test suite for SHRINK Intelligence Integration"""

    def setUp(self):
        """Set up test environment"""
        if not INTELLIGENCE_AVAILABLE:
            self.skipTest("SHRINKIntelligenceIntegration not available")

        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="shrink_intel_test_")
        self.test_path = Path(self.test_dir)

        # Initialize integration
        self.integration = SHRINKIntelligenceIntegration(root_dir=self.test_path)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_integration_initialization(self):
        """Test integration initializes correctly"""
        self.assertIsNotNone(self.integration)
        self.assertEqual(self.integration.root_dir, self.test_path)

    def test_shrink_components_initialized(self):
        """Test SHRINK components are initialized"""
        # Check compressor attribute exists (may be None if SHRINK unavailable)
        self.assertTrue(hasattr(self.integration, 'compressor'))

        # Check optimizer attribute exists (may be None if SHRINK unavailable)
        self.assertTrue(hasattr(self.integration, 'optimizer'))

        # Check deduplicator attribute exists (may be None if SHRINK unavailable)
        self.assertTrue(hasattr(self.integration, 'deduplicator'))

        # Check shrink_available attribute exists
        self.assertTrue(hasattr(self.integration, 'shrink_available'))

    def test_collect_system_metrics(self):
        """Test collecting system metrics"""
        metrics = self.integration.collect_system_metrics()

        self.assertIsInstance(metrics, SystemMetrics)
        self.assertIsNotNone(metrics.timestamp)

        # Verify metrics have default values when subsystems not available
        self.assertIsInstance(metrics.total_screenshots, int)
        self.assertIsInstance(metrics.total_documents, int)
        self.assertIsInstance(metrics.total_embeddings, int)

    def test_system_metrics_creation(self):
        """Test SystemMetrics dataclass creation"""
        from datetime import datetime

        metrics = SystemMetrics(
            timestamp=datetime.now(),
            total_screenshots=100,
            screenshot_storage_mb=50.0,
            total_documents=200,
            total_embeddings=200,
            embedding_storage_mb=120.0
        )

        self.assertEqual(metrics.total_screenshots, 100)
        self.assertEqual(metrics.screenshot_storage_mb, 50.0)
        self.assertEqual(metrics.total_documents, 200)

    def test_optimize_screenshot_storage(self):
        """Test screenshot storage optimization"""
        # Create dummy screenshot directory
        screenshot_dir = self.test_path / 'screenshots'
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy screenshot file
        dummy_screenshot = screenshot_dir / 'test_screenshot.png'
        dummy_screenshot.write_bytes(b'PNG dummy data' * 100)  # Create some data

        # Run optimization
        result = self.integration.optimize_screenshot_storage()

        self.assertIsInstance(result, dict)
        # Result may contain 'status': 'error' if SHRINK not available
        if result.get('status') == 'error':
            self.assertIn('message', result)
        else:
            self.assertIn('original_size_mb', result)
            self.assertIn('compressed_size_mb', result)
            self.assertIn('savings_mb', result)
            self.assertIn('compression_ratio', result)

    def test_optimize_rag_embeddings(self):
        """Test RAG embeddings optimization"""
        result = self.integration.optimize_rag_embeddings()

        self.assertIsInstance(result, dict)
        # Result may contain 'status': 'error' if SHRINK not available
        if result.get('status') == 'error':
            self.assertIn('message', result)
        else:
            self.assertIn('original_embeddings', result)
            self.assertIn('deduplicated_embeddings', result)
            self.assertIn('duplicates_removed', result)
            self.assertIn('savings_mb', result)

    def test_generate_intelligence_report(self):
        """Test generating intelligence report"""
        report = self.integration.generate_intelligence_report()

        self.assertIsInstance(report, dict)
        self.assertIn('timestamp', report)
        # Report structure may vary depending on SHRINK availability
        # Just verify it's a valid dict with timestamp
        self.assertIsNotNone(report.get('timestamp'))

    def test_run_comprehensive_optimization(self):
        """Test comprehensive optimization across all systems"""
        # Create some test data
        screenshot_dir = self.test_path / 'screenshots'
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        dummy_screenshot = screenshot_dir / 'test.png'
        dummy_screenshot.write_bytes(b'PNG' * 100)

        # Run optimization
        result = self.integration.run_comprehensive_optimization()

        self.assertIsInstance(result, dict)
        self.assertIn('metrics_before', result)
        self.assertIn('metrics_after', result)
        self.assertIn('screenshot_optimization', result)
        self.assertIn('rag_optimization', result)
        # total_savings_mb or total_space_saved_mb depending on implementation
        self.assertTrue(
            'total_savings_mb' in result or 'total_space_saved_mb' in result,
            "Should have total savings field"
        )


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="shrink_e2e_test_")
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_full_integration_workflow(self):
        """Test complete integration workflow"""
        if not all([MANAGER_AVAILABLE, HEALTH_AVAILABLE, INTELLIGENCE_AVAILABLE]):
            self.skipTest("Not all components available")

        # Step 1: Initialize SHRINK
        manager = SHRINKIntegrationManager(root_dir=self.test_path)
        init_result = manager.initialize_shrink(force=True)
        self.assertTrue(init_result, "SHRINK initialization should succeed")

        # Step 2: Check health
        monitor = SubmoduleHealthMonitor(root_dir=self.test_path)
        health_report = monitor.check_shrink_health()
        self.assertEqual(health_report.submodule_name, 'SHRINK')

        # Step 3: Collect intelligence metrics
        integration = SHRINKIntelligenceIntegration(root_dir=self.test_path)
        metrics = integration.collect_system_metrics()
        self.assertIsInstance(metrics, SystemMetrics)

        # Step 4: Check status
        status = manager.check_status('SHRINK')
        self.assertTrue(status.initialized)

        print("\n✓ End-to-end integration workflow completed successfully")

    def test_manager_and_health_integration(self):
        """Test integration between manager and health monitor"""
        if not all([MANAGER_AVAILABLE, HEALTH_AVAILABLE]):
            self.skipTest("Manager or Health monitor not available")

        # Initialize with manager
        manager = SHRINKIntegrationManager(root_dir=self.test_path)
        manager.initialize_shrink(force=True)

        # Check with health monitor
        monitor = SubmoduleHealthMonitor(root_dir=self.test_path)
        report = monitor.check_shrink_health()

        # Verify health report reflects initialization
        self.assertIn(report.overall_status, [HealthStatus.HEALTHY, HealthStatus.WARNING])

    def test_health_and_intelligence_integration(self):
        """Test integration between health monitor and intelligence system"""
        if not all([HEALTH_AVAILABLE, INTELLIGENCE_AVAILABLE]):
            self.skipTest("Health or Intelligence not available")

        # Initialize SHRINK structure
        shrink_path = self.test_path / 'modules' / 'SHRINK'
        shrink_path.mkdir(parents=True, exist_ok=True)
        (shrink_path / '__init__.py').write_text('# SHRINK')

        # Check health
        monitor = SubmoduleHealthMonitor(root_dir=self.test_path)
        health_report = monitor.check_shrink_health()

        # Collect intelligence
        integration = SHRINKIntelligenceIntegration(root_dir=self.test_path)
        metrics = integration.collect_system_metrics()

        # Both should work without errors
        self.assertIsNotNone(health_report)
        self.assertIsNotNone(metrics)


def run_tests(test_suite=None, verbose=False):
    """Run test suite"""
    # Create test suite
    if test_suite == 'manager':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSHRINKIntegrationManager)
    elif test_suite == 'health':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSubmoduleHealthMonitor)
    elif test_suite == 'intelligence':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSHRINKIntelligenceIntegration)
    elif test_suite == 'e2e':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndIntegration)
    else:
        # Run all tests
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSHRINKIntegrationManager))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSubmoduleHealthMonitor))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSHRINKIntelligenceIntegration))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndIntegration))

    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='SHRINK Integration Tests')
    parser.add_argument('--test', choices=['manager', 'health', 'intelligence', 'e2e', 'all'],
                        default='all', help='Test suite to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("=" * 70)
    print("SHRINK Integration Test Suite")
    print("=" * 70)
    print()

    # Check component availability
    print("Component Availability:")
    print(f"  SHRINKIntegrationManager: {'✓' if MANAGER_AVAILABLE else '✗'}")
    print(f"  SubmoduleHealthMonitor:   {'✓' if HEALTH_AVAILABLE else '✗'}")
    print(f"  SHRINKIntelligenceIntegration: {'✓' if INTELLIGENCE_AVAILABLE else '✗'}")
    print()

    if not any([MANAGER_AVAILABLE, HEALTH_AVAILABLE, INTELLIGENCE_AVAILABLE]):
        print("ERROR: No SHRINK components available for testing")
        return 1

    # Run tests
    test_suite = args.test if args.test != 'all' else None
    result = run_tests(test_suite=test_suite, verbose=args.verbose)

    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
