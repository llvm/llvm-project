"""
Tests calling builtin functions using expression evaluation.
"""



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

        # Test different builtin functions.

        self.expect_expr("__builtin_isinf(0.0f)", result_type="int", result_value="0")
        self.expect_expr("__builtin_isnormal(0.0f)", result_type="int", result_value="0")
        self.expect_expr("__builtin_constant_p(1)", result_type="int", result_value="1")
        self.expect_expr("__builtin_abs(-14)", result_type="int", result_value="14")
