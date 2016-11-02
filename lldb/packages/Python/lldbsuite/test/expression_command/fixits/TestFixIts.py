"""
Test calling an expression with errors that a FixIt can fix.
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithFixits(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @skipUnlessDarwin
    def test(self):
        """Test calling a function that throws and ObjC exception."""
        self.build()
        self.try_expressions()

    def try_expressions(self):
        """Test calling expressions with errors that can be fixed by the FixIts."""
        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex(
            'Stop here to evaluate expressions', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(True)

        frame = self.thread.GetFrameAtIndex(0)

        # Try with one error:
        value = frame.EvaluateExpression("my_pointer.first", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsUnsigned() == 10)

        # Try with two errors:
        two_error_expression = "my_pointer.second->a"
        value = frame.EvaluateExpression(two_error_expression, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
        self.assertTrue(value.GetValueAsUnsigned() == 20)

        # Now turn off the fixits, and the expression should fail:
        options.SetAutoApplyFixIts(False)
        value = frame.EvaluateExpression(two_error_expression, options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Fail())
        error_string = value.GetError().GetCString()
        self.assertTrue(
            error_string.find("fixed expression suggested:") != -1,
            "Fix was suggested")
        self.assertTrue(
            error_string.find("my_pointer->second.a") != -1,
            "Fix was right")
