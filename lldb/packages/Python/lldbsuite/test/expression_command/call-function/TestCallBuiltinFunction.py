"""
Tests calling builtin functions using expression evaluation.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandCallBuiltinFunction(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Builtins are expanded by Clang, so debug info shouldn't matter.
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Please test these expressions while stopped at this line:')

    def test(self):
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Test different builtin functions.

        interp.HandleCommand("expr __builtin_isinf(0.0f)", result)
        self.assertEqual(result.GetOutput(), "(int) $0 = 0\n")

        interp.HandleCommand("expr __builtin_isnormal(0.0f)", result)
        self.assertEqual(result.GetOutput(), "(int) $1 = 0\n")

        interp.HandleCommand("expr __builtin_constant_p(1)", result)
        self.assertEqual(result.GetOutput(), "(int) $2 = 1\n")

        interp.HandleCommand("expr __builtin_abs(-14)", result)
        self.assertEqual(result.GetOutput(), "(int) $3 = 14\n")
