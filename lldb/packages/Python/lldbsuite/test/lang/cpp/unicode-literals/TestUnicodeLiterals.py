# coding=utf8
"""
Test that the expression parser returns proper Unicode strings.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

# this test case fails because of rdar://12991846
# the expression parser does not deal correctly with Unicode expressions
# e.g.
#(lldb) expr L"Hello"
#(const wchar_t [6]) $0 = {
#  [0] = \0\0\0\0
#  [1] = \0\0\0\0
#  [2] = \0\0\0\0
#  [3] = \0\0\0\0
#  [4] = H\0\0\0
#  [5] = e\0\0\0
#}


class UnicodeLiteralsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_expr1(self):
        """Test that the expression parser returns proper Unicode strings."""
        self.build()
        self.rdar12991846(expr=1)

    def test_expr2(self):
        """Test that the expression parser returns proper Unicode strings."""
        self.build()
        self.rdar12991846(expr=2)

    def test_expr3(self):
        """Test that the expression parser returns proper Unicode strings."""
        self.build()
        self.rdar12991846(expr=3)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.source = 'main.cpp'
        self.line = line_number(
            self.source, '// Set break point at this line.')

    def rdar12991846(self, expr=None):
        """Test that the expression parser returns proper Unicode strings."""
        if self.getArchitecture() in ['i386']:
            self.skipTest(
                "Skipping because this test is known to crash on i386")

        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Break on the struct declration statement in main.cpp.
        lldbutil.run_break_set_by_file_and_line(self, "main.cpp", self.line)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.Launch() failed")

        if expr == 1:
            self.expect('expression L"hello"', substrs=['hello'])

        if expr == 2:
            self.expect('expression u"hello"', substrs=['hello'])

        if expr == 3:
            self.expect('expression U"hello"', substrs=['hello'])
