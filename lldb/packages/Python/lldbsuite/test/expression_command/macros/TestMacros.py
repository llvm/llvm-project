from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestMacros(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureClang("clang does not emit .debug_macro[.dwo] sections.")
    @expectedFailureDwo("GCC produces multiple .debug_macro.dwo sections and the spec is unclear as to what it means")
    @expectedFailureAll(hostoslist=["windows"], compiler="gcc", triple='.*-android')
    def test_expr_with_macros(self):
        self.build()

        # Get main source file
        src_file = "main.cpp"
        hdr_file = "macro1.h"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Get the path of the executable
        cwd = os.getcwd()
        exe_file = "a.out"
        exe_path  = os.path.join(cwd, exe_file)

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Set breakpoints
        bp1 = target.BreakpointCreateBySourceRegex("Break here", src_file_spec)
        self.assertTrue(bp1.IsValid() and bp1.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("MACRO_1")
        self.assertTrue(result.IsValid() and result.GetValue() == "100", "MACRO_1 = 100")

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(result.IsValid() and result.GetValue() == "200", "MACRO_2 = 200")

        result = frame.EvaluateExpression("ONE")
        self.assertTrue(result.IsValid() and result.GetValue() == "1", "ONE = 1")

        result = frame.EvaluateExpression("TWO")
        self.assertTrue(result.IsValid() and result.GetValue() == "2", "TWO = 2")

        result = frame.EvaluateExpression("THREE")
        self.assertTrue(result.IsValid() and result.GetValue() == "3", "THREE = 3")

        result = frame.EvaluateExpression("FOUR")
        self.assertTrue(result.IsValid() and result.GetValue() == "4", "FOUR = 4")

        result = frame.EvaluateExpression("HUNDRED")
        self.assertTrue(result.IsValid() and result.GetValue() == "100", "HUNDRED = 100")

        result = frame.EvaluateExpression("THOUSAND")
        self.assertTrue(result.IsValid() and result.GetValue() == "1000", "THOUSAND = 1000")

        result = frame.EvaluateExpression("MILLION")
        self.assertTrue(result.IsValid() and result.GetValue() == "1000000", "MILLION = 1000000")

        result = frame.EvaluateExpression("MAX(ONE, TWO)")
        self.assertTrue(result.IsValid() and result.GetValue() == "2", "MAX(ONE, TWO) = 2")

        result = frame.EvaluateExpression("MAX(THREE, TWO)")
        self.assertTrue(result.IsValid() and result.GetValue() == "3", "MAX(THREE, TWO) = 3")

        # Get the thread of the process
        thread.StepOver()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(result.GetError().Fail(), "Printing MACRO_2 fails in the mail file")

        result = frame.EvaluateExpression("FOUR")
        self.assertTrue(result.GetError().Fail(), "Printing FOUR fails in the main file")

        thread.StepInto()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        result = frame.EvaluateExpression("ONE")
        self.assertTrue(result.IsValid() and result.GetValue() == "1", "ONE = 1")

        result = frame.EvaluateExpression("MAX(ONE, TWO)")
        self.assertTrue(result.IsValid() and result.GetValue() == "2", "MAX(ONE, TWO) = 2")

        # This time, MACRO_1 and MACRO_2 are not visible.
        result = frame.EvaluateExpression("MACRO_1")
        self.assertTrue(result.GetError().Fail(), "Printing MACRO_1 fails in the header file")

        result = frame.EvaluateExpression("MACRO_2")
        self.assertTrue(result.GetError().Fail(), "Printing MACRO_2 fails in the header file")
