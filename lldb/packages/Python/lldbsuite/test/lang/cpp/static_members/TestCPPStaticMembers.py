"""
Tests that C++ member and static variables have correct layout and scope.
"""

from __future__ import print_function


import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPStaticMembersTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.expectedFailure  # llvm.org/pr15401
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_with_run_command(self):
        """Test that member variables have the correct layout, scope and qualifiers when stopped inside and outside C++ methods"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(line_number('main.cpp', '// breakpoint 1'))
        self.set_breakpoint(line_number('main.cpp', '// breakpoint 2'))

        self.runCmd("process launch", RUN_SUCCEEDED)
        self.expect("expression my_a.access()",
                    startstr="(long) $0 = 10")

        self.expect("expression my_a.m_a",
                    startstr="(short) $1 = 1")

        # Note: SymbolFileDWARF::ParseChildMembers doesn't call
        # AddFieldToRecordType, consistent with clang's AST layout.
        self.expect("expression my_a.s_d",
                    startstr="(int) $2 = 4")

        self.expect("expression my_a.s_b",
                    startstr="(long) $3 = 2")

        self.expect("expression A::s_b",
                    startstr="(long) $4 = 2")

        # should not be available in global scope
        self.expect("expression s_d",
                    startstr="error: use of undeclared identifier 's_d'")

        self.runCmd("process continue")
        self.expect("expression m_c",
                    startstr="(char) $5 = \'\\x03\'")

        self.expect("expression s_b",
                    startstr="(long) $6 = 2")

        self.runCmd("process continue")

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line, num_expected_locations=1, loc_exact=False)
