"""
Test basic DarwinLog functionality provided by the StructuredDataDarwinLog
plugin.

These tests are currently only supported when running against Darwin
targets.
"""

# System imports
from __future__ import print_function

import re

# LLDB imports
import lldb

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogFilterRegexMessage(darwin_log.DarwinLogEventBasedTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_accept_message_full_match(self):
        """Test that fall-through reject, accept regex whole message works."""
        log_entries = self.do_test(
            ["--no-match-accepts false",
             # Note below, the four '\' characters are to get us two
             # backslashes over on the gdb-remote side, which then
             # becomes one as the cstr interprets it as an escape
             # sequence.  This needs to be rationalized.  Initially I
             # supported std::regex ECMAScript, which has the
             # [[:digit:]] character classes and such.  That was much
             # more tenable.  The backslashes have to travel through
             # so many layers of escaping.  (And note if you take
             # off the Python raw string marker here, you need to put
             # in 8 backslashes to go to two on the remote side.)
             r'--filter "accept message regex log message sub2-cat\\\\d+"'])

        # We should have received at least one log entry.
        self.assertIsNotNone(log_entries,
                             "Log entry list should not be None.")
        self.assertEqual(len(log_entries), 1,
                         "Should receive one log entry.")
        self.assertRegexpMatches(log_entries[0]["message"], r"sub2-cat2",
                                 "First os_log call should have been skipped.")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_accept_message_partial_match(self):
        """Test that fall-through reject, accept regex message via partial
        match works."""
        log_entries = self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept message regex [^-]+2\""])

        # We should only see the second log message as we only accept
        # that message contents.
        self.assertIsNotNone(log_entries,
                             "Log entry list should not be None.")
        self.assertEqual(len(log_entries), 1,
                         "Should receive one log entry.")
        self.assertRegexpMatches(log_entries[0]["message"], r"sub2-cat2",
                                 "First os_log call should have been skipped.")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_reject_message_full_match(self):
        """Test that fall-through accept, reject regex message works."""
        log_entries = self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject message regex log message sub1-cat1\""])

        # We should only see the second log message as we rejected the first
        # via message contents rejection.
        self.assertIsNotNone(log_entries,
                             "Log entry list should not be None.")
        self.assertEqual(len(log_entries), 1,
                         "Should receive one log entry.")
        self.assertRegexpMatches(log_entries[0]["message"], r"sub2-cat2",
                                 "First os_log call should have been skipped.")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_reject_message_partial_match(self):
        """Test that fall-through accept, reject regex message by partial
        match works."""
        log_entries = self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject message regex t1\""])

        # We should only see the second log message as we rejected the first
        # via partial message contents rejection.
        self.assertIsNotNone(log_entries,
                             "Log entry list should not be None.")
        self.assertEqual(len(log_entries), 1,
                         "Should receive one log entry.")
        self.assertRegexpMatches(log_entries[0]["message"], r"sub2-cat2",
                                 "First os_log call should have been skipped.")

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(oslist=["macosx"],
                                   bugnumber="llvm.org/pr30299")
    def test_filter_accept_message_second_rule(self):
        """Test that fall-through reject, accept regex message on second rule
         works."""
        log_entries = self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept message regex non-existent\"",
             "--filter \"accept message regex cat2\""])

        # We should only see the second message since we reject by default,
        # the first filter doesn't match any, and the second filter matches
        # the message of the second log message.
        self.assertIsNotNone(log_entries,
                             "Log entry list should not be None.")
        self.assertEqual(len(log_entries), 1,
                         "Should receive one log entry.")
        self.assertRegexpMatches(log_entries[0]["message"], r"sub2-cat2",
                                 "First os_log call should have been skipped.")
