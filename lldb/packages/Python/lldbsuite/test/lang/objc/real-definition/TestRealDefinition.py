"""Test that types defined in shared libraries work correctly."""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestRealDefinition(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_frame_var_after_stop_at_interface(self):
        """Test that we can find the implementation for an objective C type"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()
        self.common_setup()

        line = line_number(
            'Foo.m', '// Set breakpoint where Bar is an interface')
        lldbutil.run_break_set_by_file_and_line(
            self, 'Foo.m', line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Run and stop at Foo
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        self.expect(
            "frame variable foo->_bar->_hidden_ivar",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(NSString *)",
                "foo->_bar->_hidden_ivar = 0x"])

    @skipUnlessDarwin
    def test_frame_var_after_stop_at_implementation(self):
        """Test that we can find the implementation for an objective C type"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires modern objc runtime")
        self.build()
        self.common_setup()

        line = line_number(
            'Bar.m', '// Set breakpoint where Bar is an implementation')
        lldbutil.run_break_set_by_file_and_line(
            self, 'Bar.m', line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Run and stop at Foo
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.runCmd("continue", RUN_SUCCEEDED)

        # Run at stop at main
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # This should display correctly.
        self.expect(
            "frame variable foo->_bar->_hidden_ivar",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                "(NSString *)",
                "foo->_bar->_hidden_ivar = 0x"])

    def common_setup(self):
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        line = line_number('main.m', '// Set breakpoint in main')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", line, num_expected_locations=1, loc_exact=True)
