"""
Test that we can call C++ template fucntions.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TemplateFunctionsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def do_test_template_function(self, add_cast):
        self.build()
        (_, _, thread, _) = lldbutil.run_to_name_breakpoint(self, "main")
        frame = thread.GetSelectedFrame()
        expr = "foo(42)"
        if add_cast:
            expr = "(int)" + expr
        expr_result = frame.EvaluateExpression(expr)
        self.assertTrue(expr_result.IsValid())
        self.assertEqual(expr_result.GetValue(), "42")

    @skipIfWindows
    def test_template_function_with_cast(self):
        self.do_test_template_function(True)

    @skipIfWindows
    @expectedFailureAll(debug_info=["dwarf", "gmodules", "dwo"])
    def test_template_function_without_cast(self):
        self.do_test_template_function(False)
