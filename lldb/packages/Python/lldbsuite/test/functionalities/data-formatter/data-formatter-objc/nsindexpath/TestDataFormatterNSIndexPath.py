# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import datetime
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NSIndexPathDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def appkit_tester_impl(self, commands):
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        commands()

    @skipUnlessDarwin
    def test_nsindexpath_with_run_command(self):
        """Test formatters for NSIndexPath."""
        self.appkit_tester_impl(self.nsindexpath_data_formatter_commands)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// break here')

    def nsindexpath_data_formatter_commands(self):
        # check 'frame variable'
        self.expect(
            'frame variable --ptr-depth=1 -d run -- indexPath1',
            substrs=['[0] = 1'])
        self.expect(
            'frame variable --ptr-depth=1 -d run -- indexPath2',
            substrs=[
                '[0] = 1',
                '[1] = 2'])
        self.expect(
            'frame variable --ptr-depth=1 -d run -- indexPath3',
            substrs=[
                '[0] = 1',
                '[1] = 2',
                '[2] = 3'])
        self.expect(
            'frame variable --ptr-depth=1 -d run -- indexPath4',
            substrs=[
                '[0] = 1',
                '[1] = 2',
                '[2] = 3',
                '[3] = 4'])
        self.expect(
            'frame variable --ptr-depth=1 -d run -- indexPath5',
            substrs=[
                '[0] = 1',
                '[1] = 2',
                '[2] = 3',
                '[3] = 4',
                '[4] = 5'])

        # and 'expression'
        self.expect(
            'expression --ptr-depth=1 -d run -- indexPath1',
            substrs=['[0] = 1'])
        self.expect(
            'expression --ptr-depth=1 -d run -- indexPath2',
            substrs=[
                '[0] = 1',
                '[1] = 2'])
        self.expect(
            'expression --ptr-depth=1 -d run -- indexPath3',
            substrs=[
                '[0] = 1',
                '[1] = 2',
                '[2] = 3'])
        self.expect('expression --ptr-depth=1 -d run -- indexPath4',
                    substrs=['[0] = 1', '[1] = 2', '[2] = 3', '[3] = 4'])
        self.expect(
            'expression --ptr-depth=1 -d run -- indexPath5',
            substrs=[
                '[0] = 1',
                '[1] = 2',
                '[2] = 3',
                '[3] = 4',
                '[4] = 5'])
