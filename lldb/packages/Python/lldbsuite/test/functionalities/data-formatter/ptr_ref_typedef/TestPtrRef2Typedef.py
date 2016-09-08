"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class PtrRef2TypedefTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set breakpoint here')

    def test_with_run_command(self):
        """Test that a pointer/reference to a typedef is formatted as we want."""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

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

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd('type summary add --cascade true -s "IntPointer" "int *"')
        self.runCmd('type summary add --cascade true -s "IntLRef" "int &"')
        self.runCmd('type summary add --cascade true -s "IntRRef" "int &&"')

        self.expect(
            "frame variable x",
            substrs=[
                '(Foo *) x = 0x',
                'IntPointer'])
        # note: Ubuntu 12.04 x86_64 build with gcc 4.8.2 is getting a
        # const after the ref that isn't showing up on FreeBSD. This
        # tweak changes the behavior so that the const is not part of
        # the match.
        self.expect(
            "frame variable y", substrs=[
                '(Foo &', ') y = 0x', 'IntLRef'])
        self.expect(
            "frame variable z", substrs=[
                '(Foo &&', ') z = 0x', 'IntRRef'])
