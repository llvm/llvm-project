"""
Test calling user defined functions using expression evaluation.

Note:
  LLDBs current first choice of evaluating functions is using the IR interpreter,
  which is only supported on Hexagon. Otherwise JIT is used for the evaluation.

"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprCommandCallUserDefinedFunction(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.cpp',
                                '// Please test these expressions while stopped at this line:')
    @expectedFlakeyDsym("llvm.org/pr20274")
    @expectedFailureWindows("llvm.org/pr24489: Name lookup not working correctly on Windows")
    def test(self):
        """Test return values of user defined function calls."""
        self.build()

        # Set breakpoint in main and run exe
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # Test recursive function call.
        self.expect("expr fib(5)", substrs = ['$0 = 5'])

        # Test function with more than one paramter
        self.expect("expr add(4,8)", substrs = ['$1 = 12'])

        # Test nesting function calls in function paramters
        self.expect("expr add(add(5,2),add(3,4))", substrs = ['$2 = 14'])
        self.expect("expr add(add(5,2),fib(5))", substrs = ['$3 = 12'])

        # Test function with pointer paramter
        self.expect("exp stringCompare((const char*) \"Hello world\")", substrs = ['$4 = true'])
        self.expect("exp stringCompare((const char*) \"Hellworld\")", substrs = ['$5 = false'])
