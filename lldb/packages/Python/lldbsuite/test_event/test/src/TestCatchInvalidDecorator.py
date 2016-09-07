#!/usr/bin/env python
"""
Tests that the event system reports issues during decorator
handling as errors.
"""
# System-provided imports
import os
import unittest

# Local-provided imports
import event_collector


class TestCatchInvalidDecorator(unittest.TestCase):

    TEST_DIR = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "resources",
        "invalid_decorator")

    def test_with_whole_file(self):
        """
        Test that a non-existent decorator generates a test-event error
        when running all tests in the file.
        """
        # Determine the test case file we're using.
        test_file = os.path.join(self.TEST_DIR, "TestInvalidDecorator.py")

        # Collect all test events generated for this file.
        error_results = _filter_error_results(
            event_collector.collect_events_whole_file(test_file))

        self.assertGreater(
            len(error_results),
            0,
            "At least one job or test error result should have been returned")

    def test_with_function_filter(self):
        """
        Test that a non-existent decorator generates a test-event error
        when running a filtered test.
        """
        # Collect all test events generated during running of tests
        # in a given directory using a test name filter.  Internally,
        # this runs through a different code path that needs to be
        # set up to catch exceptions.
        error_results = _filter_error_results(
            event_collector.collect_events_for_directory_with_filter(
                self.TEST_DIR,
                "NonExistentDecoratorTestCase.test"))

        self.assertGreater(
            len(error_results),
            0,
            "At least one job or test error result should have been returned")


def _filter_error_results(events):
    # Filter out job result events.
    return [
        event
        for event in events
        if event.get("event", None) in ["job_result", "test_result"] and
        event.get("status", None) == "error"
    ]


if __name__ == "__main__":
    unittest.main()
