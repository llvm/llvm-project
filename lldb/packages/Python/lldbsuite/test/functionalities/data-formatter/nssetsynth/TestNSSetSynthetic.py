"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import datetime
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NSSetSyntheticTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    @skipUnlessDarwin
    def test_rdar12529957_with_run_command(self):
        """Test that NSSet reports its synthetic children properly."""
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

        # Now check that we are displaying Cocoa classes correctly
        self.expect('frame variable set',
                    substrs=['4 elements'])
        self.expect('frame variable mutable',
                    substrs=['9 elements'])
        self.expect(
            'frame variable set --ptr-depth 1 -d run -T',
            substrs=[
                '4 elements',
                '[0]',
                '[1]',
                '[2]',
                '[3]',
                'hello',
                'world',
                '(int)1',
                '(int)2'])
        self.expect(
            'frame variable mutable --ptr-depth 1 -d run -T',
            substrs=[
                '9 elements',
                '(int)5',
                '@"3 elements"',
                '@"www.apple.com"',
                '(int)3',
                '@"world"',
                '(int)4'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs=['0 elements'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs=['4 elements'])
        self.expect(
            'frame variable mutable --ptr-depth 1 -d run -T',
            substrs=[
                '4 elements',
                '[0]',
                '[1]',
                '[2]',
                '[3]',
                'hello',
                'world',
                '(int)1',
                '(int)2'])

        self.runCmd("next")
        self.expect('frame variable mutable',
                    substrs=['4 elements'])
        self.expect(
            'frame variable mutable --ptr-depth 1 -d run -T',
            substrs=[
                '4 elements',
                '[0]',
                '[1]',
                '[2]',
                '[3]',
                'hello',
                'world',
                '(int)1',
                '(int)2'])
