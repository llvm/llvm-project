"""
Test calling user defined functions using expression evaluation.
This test checks that typesystem lookup works correctly for typedefs of
untagged structures.

Ticket: https://llvm.org/bugs/show_bug.cgi?id=26790
"""

from __future__ import print_function

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExprLookupAnonStructTypedef(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        # Find the breakpoint
        self.line = line_number('main.cpp', '// lldb testsuite break')

    @expectedFailureAll(
        oslist=['linux'],
        archs=['arm'],
        bugnumber="llvm.org/pr27868")
    def test(self):
        """Test typedeffed untagged struct arguments for function call expressions"""
        self.build()

        self.runCmd("file "+self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.cpp",
            self.line,
            num_expected_locations=-1,
            loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("expr multiply(&s)", substrs=['$0 = 1'])
