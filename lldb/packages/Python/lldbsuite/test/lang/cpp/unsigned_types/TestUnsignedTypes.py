"""
Test that variables with unsigned types display correctly.
"""

from __future__ import print_function


import os
import time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class UnsignedTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test(self):
        """Test that variables with unsigned types display correctly."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # GCC puts a breakpoint on the last line of a multi-line expression, so
        # if GCC is the target compiler, we cannot rely on an exact line match.
        need_exact = "gcc" not in self.getCompiler()
        # Break on line 19 in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.line,
            num_expected_locations=-1,
            loc_exact=need_exact)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Test that unsigned types display correctly.
        self.expect(
            "frame variable --show-types --no-args",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\((short unsigned int|unsigned short)\) the_unsigned_short = 99"],
            substrs=[
                "(unsigned char) the_unsigned_char = 'c'",
                "(unsigned int) the_unsigned_int = 99",
                "(unsigned long) the_unsigned_long = 99",
                "(unsigned long long) the_unsigned_long_long = 99",
                "(uint32_t) the_uint32 = 99"])
