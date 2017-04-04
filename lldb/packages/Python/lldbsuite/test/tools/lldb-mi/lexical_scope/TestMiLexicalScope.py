"""
Test lldb-mi -stack-list-locals -stack-list-variables and -var-create commands
for variables with the same name in sibling lexical scopes.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiLexicalScopeTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    def test_lldbmi_var_create_in_sibling_scope(self):
        """Test that 'lldb-mi --interpreter' works with sibling lexical scopes."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Breakpoint inside first scope
        line = line_number('main.cpp', '// BP_first')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"\d+\"")

        # Breakpoint inside second scope
        line = line_number('main.cpp', '// BP_second')
        self.runCmd("-break-insert --file main.cpp:%d" % line)
        self.expect("\^done,bkpt={number=\"\d+\"")

        # Run to the first scope
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check that only variables a and b exist with expected values
        self.runCmd("-stack-list-locals --thread 1 --frame 0 --all-values")
        self.expect("\^done,locals=\[{name=\"a\",value=\"1\"},{name=\"b\",value=\"2\"}\]")

        # Create variable object for local variable b
        self.runCmd("-var-create - * \"b\"")
        self.expect(
            "\^done,name=\"var\d+\",numchild=\"0\",value=\"2\",type=\"int\",thread-id=\"1\",has_more=\"0\"")

        # Run to the second scope
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check that only variables a and b exist with expected values,
        # but variable b is different from previous breakpoint
        self.runCmd("-stack-list-variables --thread 1 --frame 0 --all-values")
        self.expect("\^done,variables=\[{name=\"a\",value=\"1\"},{name=\"b\",value=\"3\"}\]")

        # Create variable object for local variable b
        self.runCmd("-var-create - * \"b\"")
        self.expect(
            "\^done,name=\"var\d+\",numchild=\"0\",value=\"3\",type=\"short\",thread-id=\"1\",has_more=\"0\"")
