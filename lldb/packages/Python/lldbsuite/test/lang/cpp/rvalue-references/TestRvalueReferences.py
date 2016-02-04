"""
Tests that rvalue references are supported in C++
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class RvalueReferencesTestCase(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)
    
    #rdar://problem/11479676
    @expectedFailureIcc("ICC (13.1, 14-beta) do not emit DW_TAG_rvalue_reference_type.")
    @expectedFailureWindows("llvm.org/pr24489: Name lookup not working correctly on Windows")
    def test_with_run_command(self):
        """Test that rvalues are supported in the C++ expression parser"""
        self.build()
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.cpp', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.cpp', '// breakpoint 2'))

        self.runCmd("process launch", RUN_SUCCEEDED)

        # Note that clang as of r187480 doesn't emit DW_TAG_const_type, unlike gcc 4.8.1
        # With gcc 4.8.1, lldb reports the type as (int &&const)
        self.expect("frame variable i",
                    startstr = "(int &&",
                    substrs = ["i = 0x", "&i = 3"])

        self.expect("expression -- i",
                    startstr = "(int) ",
                    substrs = ["3"])

        self.expect("breakpoint delete 1")

        self.runCmd("process continue")
        
        self.expect("expression -- foo(2)")

        self.expect("expression -- int &&j = 3; foo(j)",
                    error = True)

        self.expect("expression -- int &&k = 6; k",
                    startstr = "(int) $1 = 6")

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", line, num_expected_locations=1, loc_exact=True)
