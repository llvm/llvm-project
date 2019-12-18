"""
Test basic DarwinLog functionality provided by the StructuredDataDarwinLog
plugin.

These tests are currently only supported when running against Darwin
targets.
"""


import lldb

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogFilterMatchCategory(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogFilterMatchCategory, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = self.getBuildArtifact("a.out")
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(TestDarwinLogFilterMatchCategory, self).tearDown()

    # ==========================================================================
    # category filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    def test_filter_accept_category_full_match(self):
        """Test that fall-through reject, accept match single category works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept category match cat2\""]
        )

        # We should only see the second log message as we only accept
        # that category.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_category_partial_match(self):
        """Test that fall-through reject, accept regex category via partial match works."""
        self.do_test(
            ["--no-match-accepts false",
             # Fully match the second message.
             "--filter \"accept category match cat2\"",
             "--filter \"accept category match at1\""]   # Only partially match first message.  Should not show up.
        )

        # We should only see the second log message as we only accept
        # that category.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_category_full_match(self):
        """Test that fall-through accept, reject match category works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject category match cat1\""]
        )

        # We should only see the second log message as we rejected the first
        # via category rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_category_second_rule(self):
        """Test that fall-through reject, accept match category on second rule works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept category match non-existent\"",
             "--filter \"accept category match cat2\""
             ]
        )

        # We should only see the second message since we reject by default,
        # the first filter doesn't match any, and the second filter matches
        # the category of the second log message.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")
