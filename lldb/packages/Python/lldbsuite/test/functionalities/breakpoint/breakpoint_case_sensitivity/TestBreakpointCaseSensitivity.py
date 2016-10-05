"""
Test case sensitivity of paths on Windows / POSIX
llvm.org/pr22667
"""

import os
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbplatform, lldbplatformutil


class BreakpointCaseSensitivityTestCase(TestBase):
    mydir = TestBase.compute_mydir(__file__)
    BREAKPOINT_TEXT = 'Set a breakpoint here'

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.line = line_number('main.c', self.BREAKPOINT_TEXT)

    @skipIf(hostoslist=no_match(['windows']))  # Skip for non-windows platforms
    def test_breakpoint_matches_file_with_different_case(self):
        """Set breakpoint on file, should match files with different case on Windows"""
        self.build()
        self.case_sensitivity_breakpoint(True)

    @skipIf(hostoslist=['windows'])  # Skip for windows platforms
    def test_breakpoint_doesnt_match_file_with_different_case(self):
        """Set breakpoint on file, shouldn't match files with different case on POSIX systems"""
        self.build()
        self.case_sensitivity_breakpoint(False)

    def case_sensitivity_breakpoint(self, case_insensitive):
        """Set breakpoint on file, should match files with different case if case_insensitive is True"""

        # use different case to check CreateTarget
        exe = 'a.out'
        if case_insensitive:
            exe = exe.upper()

        exe = os.path.join(os.getcwd(), exe)

        # Create a target by the debugger.
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)
        cwd = os.getcwd()

        # try both BreakpointCreateByLocation and BreakpointCreateBySourceRegex
        for regex in [False, True]:
            # should always hit
            self.check_breakpoint('main.c', regex, True)
            # should always hit
            self.check_breakpoint(os.path.join(cwd, 'main.c'), regex, True)
            # different case for directory
            self.check_breakpoint(os.path.join(cwd.upper(), 'main.c'),
                                  regex,
                                  case_insensitive)
            # different case for file
            self.check_breakpoint('Main.c',
                                  regex,
                                  case_insensitive)
            # different case for both
            self.check_breakpoint(os.path.join(cwd.upper(), 'Main.c'),
                                  regex,
                                  case_insensitive)

    def check_breakpoint(self, file, source_regex, should_hit):
        """
        Check breakpoint hit at given file set by given method

        file:
            File where insert the breakpoint

        source_regex:
            True for testing using BreakpointCreateBySourceRegex,
            False for  BreakpointCreateByLocation

        should_hit:
            True if the breakpoint should hit, False otherwise
        """

        desc = ' file %s set by %s' % (
            file, 'regex' if source_regex else 'location')
        if source_regex:
            breakpoint = self.target.BreakpointCreateBySourceRegex(
                self.BREAKPOINT_TEXT, lldb.SBFileSpec(file))
        else:
            breakpoint = self.target.BreakpointCreateByLocation(
                file, self.line)

        self.assertEqual(breakpoint and breakpoint.GetNumLocations() == 1,
                         should_hit,
                         VALID_BREAKPOINT + desc)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)

        self.assertEqual(location.IsValid(),
                         should_hit,
                         VALID_BREAKPOINT_LOCATION + desc)

        process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID + desc)

        if should_hit:
            # Did we hit our breakpoint?
            from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint
            threads = get_threads_stopped_at_breakpoint(process, breakpoint)
            self.assertEqual(
                len(threads),
                1,
                "There should be a thread stopped at breakpoint" +
                desc)
            # The hit count for the breakpoint should be 1.
            self.assertEqual(breakpoint.GetHitCount(), 1)

        else:
            # check the breakpoint was not hit
            self.assertEqual(lldb.eStateExited, process.GetState())
            self.assertEqual(breakpoint.GetHitCount(), 0)

        # let process finish
        process.Continue()

        # cleanup
        self.target.BreakpointDelete(breakpoint.GetID())
