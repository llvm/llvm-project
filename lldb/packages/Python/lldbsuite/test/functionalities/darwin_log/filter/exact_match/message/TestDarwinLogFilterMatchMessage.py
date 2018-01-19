"""
Test basic DarwinLog functionality provided by the StructuredDataDarwinLog
plugin.

These tests are currently only supported when running against Darwin
targets.
"""

from __future__ import print_function

import lldb
import os
import re

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogFilterMatchMessage(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogFilterMatchMessage, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = self.getBuildArtifact("a.out")
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

        self.strict_sources = True

        # Turn on process monitor logging while we work out issues.
        self.enable_process_monitor_logging = True

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(TestDarwinLogFilterMatchMessage, self).tearDown()

    # ==========================================================================
    # category filter tests
    # ==========================================================================

    EXPECT_REGEXES = [
        re.compile(r"log message ([^-]+)-(\S+)"),
        re.compile(r"exited with status")
    ]

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_accept_message_full_match(self):
        """Test that fall-through reject, accept match whole message works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept message match log message sub2-cat2\""],
            expect_regexes=self.EXPECT_REGEXES
        )

        # We should only see the second log message as we only accept
        # that message contents.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_no_accept_message_partial_match(self):
        """Test that fall-through reject, match message via partial content match doesn't accept."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept message match log message sub2-cat2\"",
             "--filter \"accept message match sub1-cat1\""],
            expect_regexes=self.EXPECT_REGEXES
        )

        # We should only see the second log message as the partial match on
        # the first message should not pass.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_reject_category_full_match(self):
        """Test that fall-through accept, reject match message works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject message match log message sub1-cat1\""],
            expect_regexes=self.EXPECT_REGEXES
        )

        # We should only see the second log message as we rejected the first
        # via message contents rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_accept_category_second_rule(self):
        """Test that fall-through reject, accept match category on second rule works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept message match non-existent\"",
             "--filter \"accept message match log message sub2-cat2\""],
            expect_regexes=self.EXPECT_REGEXES
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
