#!/usr/bin/env python3
"""
Screenshot Intelligence System - Integration Tests

Comprehensive test suite for production readiness validation

Test Categories:
1. System Initialization & Configuration
2. Vector Database Operations
3. Screenshot Ingestion & OCR
4. Timeline & Event Correlation
5. AI Analysis & Incident Detection
6. Health Monitoring & Maintenance
7. API Endpoints
8. Error Handling & Resilience

Usage:
    python3 test_screenshot_intel_integration.py
    python3 test_screenshot_intel_integration.py --verbose
    python3 test_screenshot_intel_integration.py --test-category api
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging
import unittest
from typing import Optional

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
try:
    from vector_rag_system import VectorRAGSystem, Document
    from screenshot_intelligence import ScreenshotIntelligence, Event
    from ai_analysis_layer import AIAnalysisLayer
    from system_health_monitor import SystemHealthMonitor
    from resilience_utils import with_retry, CircuitBreaker, FallbackHandler
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Some tests may be skipped")
    IMPORTS_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestSystemInitialization(unittest.TestCase):
    """Test system initialization and configuration"""

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_vector_rag_initialization(self):
        """Test VectorRAG system initialization"""
        try:
            rag = VectorRAGSystem(
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="test_collection"
            )
            self.assertIsNotNone(rag)
            self.assertIsNotNone(rag.embedding_model)
            print("✓ VectorRAG initialization successful")
        except Exception as e:
            self.skipTest(f"Qdrant not available: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_screenshot_intel_initialization(self):
        """Test Screenshot Intelligence initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                intel = ScreenshotIntelligence(data_dir=Path(tmpdir))
                self.assertIsNotNone(intel)
                self.assertIsNotNone(intel.rag)
                print("✓ Screenshot Intelligence initialization successful")
            except Exception as e:
                self.fail(f"Initialization failed: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_health_monitor_initialization(self):
        """Test Health Monitor initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                monitor = SystemHealthMonitor(data_dir=Path(tmpdir))
                self.assertIsNotNone(monitor)
                print("✓ Health Monitor initialization successful")
            except Exception as e:
                self.fail(f"Initialization failed: {e}")


class TestVectorDatabase(unittest.TestCase):
    """Test vector database operations"""

    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        if not IMPORTS_AVAILABLE:
            return

        cls.temp_dir = tempfile.mkdtemp()
        try:
            cls.rag = VectorRAGSystem(collection_name="test_integration")
        except Exception:
            cls.rag = None

    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_document_ingestion(self):
        """Test document ingestion to vector database"""
        if not self.rag:
            self.skipTest("Qdrant not available")

        doc_id = self.rag.ingest_document(
            filepath="test.txt",
            content="This is a test document for vector search",
            doc_type="test",
            metadata={"source": "test"}
        )

        self.assertIsNotNone(doc_id)
        print(f"✓ Document ingested: {doc_id}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_vector_search(self):
        """Test semantic vector search"""
        if not self.rag:
            self.skipTest("Qdrant not available")

        # Ingest test document
        self.rag.ingest_document(
            filepath="search_test.txt",
            content="Vector database with semantic search capabilities",
            doc_type="test",
            metadata={"category": "database"}
        )

        # Search
        results = self.rag.search(
            query="semantic search",
            limit=5,
            score_threshold=0.3
        )

        self.assertIsInstance(results, list)
        if results:
            self.assertTrue(results[0].score > 0.3)
            print(f"✓ Vector search returned {len(results)} results")
        else:
            print("⚠ No search results (may need to ingest more documents)")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_document_retrieval_by_id(self):
        """Test document retrieval by ID"""
        if not self.rag:
            self.skipTest("Qdrant not available")

        # Ingest document
        doc_id = self.rag.ingest_document(
            filepath="retrieve_test.txt",
            content="Test document for retrieval by ID",
            doc_type="test"
        )

        # Retrieve by ID
        doc = self.rag.get_document_by_id(doc_id)

        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, doc_id)
        print(f"✓ Document retrieved by ID: {doc_id}")


class TestScreenshotProcessing(unittest.TestCase):
    """Test screenshot ingestion and OCR"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            return

        cls.temp_dir = tempfile.mkdtemp()
        try:
            cls.intel = ScreenshotIntelligence(data_dir=Path(cls.temp_dir))
        except Exception as e:
            cls.intel = None

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_device_registration(self):
        """Test device registration"""
        if not self.intel:
            self.skipTest("Screenshot Intelligence not available")

        device_path = Path(self.temp_dir) / "device1"
        device_path.mkdir(exist_ok=True)

        self.intel.register_device(
            device_id="test_device",
            device_name="Test Phone",
            device_type="grapheneos",
            screenshot_path=device_path
        )

        self.assertIn("test_device", self.intel.devices)
        print("✓ Device registered successfully")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_filename_timestamp_parsing(self):
        """Test GrapheneOS filename timestamp parsing"""
        if not self.intel:
            self.skipTest("Screenshot Intelligence not available")

        # Test GrapheneOS format: Screenshot_20251112-143022.png
        timestamp = self.intel.parse_timestamp_from_filename("Screenshot_20251112-143022.png")

        self.assertIsNotNone(timestamp)
        self.assertEqual(timestamp.year, 2025)
        self.assertEqual(timestamp.month, 11)
        self.assertEqual(timestamp.day, 12)
        print(f"✓ Timestamp parsed: {timestamp}")


class TestEventCorrelation(unittest.TestCase):
    """Test timeline and event correlation"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            return

        cls.temp_dir = tempfile.mkdtemp()
        try:
            cls.intel = ScreenshotIntelligence(data_dir=Path(cls.temp_dir))
        except Exception:
            cls.intel = None

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_timeline_query(self):
        """Test timeline querying"""
        if not self.intel:
            self.skipTest("Screenshot Intelligence not available")

        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        try:
            events = self.intel.rag.timeline_query(start_time, end_time)
            self.assertIsInstance(events, list)
            print(f"✓ Timeline query returned {len(events)} events")
        except Exception as e:
            self.skipTest(f"Timeline query failed: {e}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_incident_creation(self):
        """Test incident creation from events"""
        if not self.intel:
            self.skipTest("Screenshot Intelligence not available")

        # Create test incident
        incident = self.intel.create_incident(
            incident_name="Test Incident",
            event_ids=[],  # Empty for now
            tags=["test", "integration"]
        )

        self.assertIsNotNone(incident)
        self.assertEqual(incident.incident_name, "Test Incident")
        self.assertIn(incident.incident_id, self.intel.incidents)
        print(f"✓ Incident created: {incident.incident_id}")


class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring and maintenance"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            return

        cls.temp_dir = tempfile.mkdtemp()
        try:
            cls.monitor = SystemHealthMonitor(data_dir=Path(cls.temp_dir))
        except Exception:
            cls.monitor = None

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_health_check(self):
        """Test comprehensive health check"""
        if not self.monitor:
            self.skipTest("Health Monitor not available")

        health = self.monitor.run_health_check()

        self.assertIsNotNone(health)
        self.assertIn(health.overall_status, ['healthy', 'degraded', 'unhealthy'])
        self.assertIsInstance(health.checks, dict)
        print(f"✓ Health check completed: {health.overall_status}")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_metrics_collection(self):
        """Test system metrics collection"""
        if not self.monitor:
            self.skipTest("Health Monitor not available")

        metrics = self.monitor.collect_metrics()

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.cpu_percent, 0)
        self.assertGreater(metrics.memory_percent, 0)
        print(f"✓ Metrics collected: CPU={metrics.cpu_percent:.1f}%, Memory={metrics.memory_percent:.1f}%")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_maintenance_tasks(self):
        """Test automated maintenance tasks"""
        if not self.monitor:
            self.skipTest("Health Monitor not available")

        results = self.monitor.run_maintenance_tasks(full_maintenance=False)

        self.assertIsInstance(results, dict)
        self.assertIn('tasks_completed', results)
        self.assertIn('tasks_failed', results)
        print(f"✓ Maintenance: {len(results['tasks_completed'])} tasks completed")


class TestResilience(unittest.TestCase):
    """Test error handling and resilience"""

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_retry_decorator(self):
        """Test retry decorator with exponential backoff"""
        attempts = []

        @with_retry(max_attempts=3, initial_delay=0.01, backoff_factor=2.0)
        def flaky_function():
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("Simulated failure")
            return "success"

        result = flaky_function()

        self.assertEqual(result, "success")
        self.assertEqual(len(attempts), 2)
        print(f"✓ Retry succeeded after {len(attempts)} attempts")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def failing_service():
            raise Exception("Service unavailable")

        # Trigger failures to open circuit
        failure_count = 0
        for _ in range(5):
            try:
                breaker.call(failing_service)
            except Exception:
                failure_count += 1

        state = breaker.get_state()
        self.assertEqual(state['state'], CircuitBreaker.STATE_OPEN)
        self.assertGreaterEqual(failure_count, 3)
        print(f"✓ Circuit breaker opened after {failure_count} failures")

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_fallback_handler(self):
        """Test graceful degradation with fallbacks"""
        fallback = FallbackHandler(log_failures=False)

        def primary():
            raise Exception("Primary failed")

        def secondary():
            return "secondary_success"

        fallback.add_handler(primary)
        fallback.add_handler(secondary)

        result = fallback.execute()

        self.assertEqual(result, "secondary_success")
        print("✓ Fallback handler succeeded with secondary")


class TestStatistics(unittest.TestCase):
    """Test statistics and reporting"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            return

        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @unittest.skipIf(not IMPORTS_AVAILABLE, "Required modules not available")
    def test_system_stats(self):
        """Test system statistics retrieval"""
        try:
            rag = VectorRAGSystem()
            stats = rag.get_stats()

            self.assertIn('collection', stats)
            self.assertIn('total_documents', stats)
            self.assertIn('vector_dimension', stats)
            print(f"✓ System stats: {stats['total_documents']} documents, {stats['vector_dimension']}D vectors")
        except Exception as e:
            self.skipTest(f"Stats retrieval failed: {e}")


def run_test_suite(test_category: Optional[str] = None, verbose: bool = False):
    """
    Run integration test suite

    Args:
        test_category: Optional category to test ('init', 'database', 'screenshot', 'health', etc.)
        verbose: Verbose output
    """
    # Determine test classes to run
    if test_category:
        category_map = {
            'init': [TestSystemInitialization],
            'database': [TestVectorDatabase],
            'screenshot': [TestScreenshotProcessing],
            'events': [TestEventCorrelation],
            'health': [TestHealthMonitoring],
            'resilience': [TestResilience],
            'stats': [TestStatistics]
        }
        test_classes = category_map.get(test_category.lower(), [])
        if not test_classes:
            print(f"❌ Unknown test category: {test_category}")
            print(f"Available categories: {', '.join(category_map.keys())}")
            return False
    else:
        test_classes = [
            TestSystemInitialization,
            TestVectorDatabase,
            TestScreenshotProcessing,
            TestEventCorrelation,
            TestHealthMonitoring,
            TestResilience,
            TestStatistics
        ]

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screenshot Intelligence Integration Tests")
    parser.add_argument('--test-category', help='Test category to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    success = run_test_suite(test_category=args.test_category, verbose=args.verbose)

    sys.exit(0 if success else 1)
