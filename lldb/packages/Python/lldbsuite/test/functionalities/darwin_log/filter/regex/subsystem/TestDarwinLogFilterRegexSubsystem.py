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


class TestDarwinLogFilterRegexSubsystem(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogFilterRegexSubsystem, self).setUp()

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
        super(TestDarwinLogFilterRegexSubsystem, self).tearDown()

    # ==========================================================================
    # basic filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    def test_fallthrough_reject(self):
        """Test that a single fall-through reject regex rule rejects all logging."""
        self.do_test(
            ["--no-match-accepts false"]
        )

        # We should not match any log lines.
        self.assertIsNotNone(self.child.match)
        self.assertFalse((len(self.child.match.groups()) > 0) and
                         (self.child.match.group(1) in ["sub1", "sub2"]),
                         "log line should not have been received")

    # ==========================================================================
    # subsystem filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    def test_filter_accept_subsystem_full_match(self):
        """Test that fall-through reject, accept regex single subsystem works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept subsystem regex org.llvm.lldb.test.sub2\""]
        )

        # We should only see the second log message as we only accept
        # that subsystem.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 0) and (
                self.child.match.group(1) == "sub2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_subsystem_partial_match(self):
        """Test that fall-through reject, accept regex subsystem via partial-match works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept subsystem regex org.llvm.+.sub2\""]
        )

        # We should only see the second log message as we only accept
        # that subsystem.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 0) and (
                self.child.match.group(1) == "sub2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_subsystem_full_match(self):
        """Test that fall-through accept, reject regex subsystem works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject subsystem regex org.llvm.lldb.test.sub1\""]
        )

        # We should only see the second log message as we rejected the first
        # via subsystem rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 0) and (
                self.child.match.group(1) == "sub2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_subsystem_partial_match(self):
        """Test that fall-through accept, reject regex subsystem by partial match works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject subsystem regex org.*sub1\""]
        )

        # We should only see the second log message as we rejected the first
        # via subsystem rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 0) and (
                self.child.match.group(1) == "sub2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_subsystem_second_rule(self):
        """Test that fall-through reject, accept regex subsystem on second rule works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept subsystem regex non-existent\"",
             "--filter \"accept subsystem regex org.llvm.lldb.test.sub2\""
             ]
        )

        # We should only see the second message since we reject by default,
        # the first filter doesn't match any, and the second filter matches
        # the subsystem of the second log message.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 0) and (
                self.child.match.group(1) == "sub2"),
            "first log line should not be present, second log line "
            "should be")
