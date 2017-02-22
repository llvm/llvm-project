"""
Test basic DarwinLog functionality provided by the StructuredDataDarwinLog
plugin.

These tests are currently only supported when running against Darwin
targets.
"""

# System imports
from __future__ import print_function

# LLDB imports
from lldbsuite.test import darwin_log
from lldbsuite.test import decorators
from lldbsuite.test import lldbtest


class TestDarwinLogBasic(darwin_log.DarwinLogEventBasedTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.add_test_categories(['pyapi'])
    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(archs=["i386"], bugnumber="rdar://28655626")
    @decorators.expectedFailureAll(bugnumber="rdar://30645203")
    def test_SBStructuredData_gets_broadcasted(self):
        """Exercise SBStructuredData API."""

        # Run the test.
        log_entries = self.do_test(None, max_entry_count=2)

        # Validate that we received our two log entries.
        self.assertEqual(len(log_entries), 1,
                         "Expected one log entry to arrive via events.")
        self.assertEqual(log_entries[0]['message'], "Hello, world",
                         "Log message should match expected content.")
