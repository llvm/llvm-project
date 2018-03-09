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


class TestDarwinLogFilterRegexActivityChain(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogFilterRegexActivityChain, self).setUp()

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
        super(TestDarwinLogFilterRegexActivityChain, self).tearDown()

    # ==========================================================================
    # activity-chain filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_chain_full_match(self):
        """Test that fall-through reject, accept full-match activity chain works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity-chain regex "
             "parent-activity:child-activity\""])

        # We should only see the second log message as we only accept
        # that activity.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_chain_partial_match(self):
        """Test that fall-through reject, accept activity-chain via partial match works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity-chain regex :child-activity\""])

        # We should only see the second log message as we only accept
        # that activity.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_activity_chain_full_match(self):
        """Test that fall-through accept, reject activity-chain works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject activity-chain regex parent-activity:child-..tivity\""])

        # We should only see the second log message as we rejected the first
        # via activity-chain rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat1"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_activity_chain_partial_match(self):
        """Test that fall-through accept, reject activity-chain by partial match works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject activity-chain regex ^p[^:]+$\""])

        # We should only see the second log message as we rejected the first
        # via activity-chain rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_chain_second_rule(self):
        """Test that fall-through reject, accept activity-chain on second rule works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity-chain regex non-existent\"",
             "--filter \"accept activity-chain regex child-activity\""])

        # We should only see the second message since we reject by default,
        # the first filter doesn't match any, and the second filter matches
        # the activity-chain of the second log message.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")
