"""
Test that global operators are found and evaluated.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppGlobalOperators(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_with_run_command(self):
        self.build()

        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Get the path of the executable
        cwd = os.getcwd()
        exe_file = "a.out"
        exe_path = os.path.join(cwd, exe_file)

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        main_breakpoint = target.BreakpointCreateBySourceRegex(
            "// break here", src_file_spec)
        self.assertTrue(
            main_breakpoint.IsValid() and main_breakpoint.GetNumLocations() >= 1,
            VALID_BREAKPOINT)

        # Launch the process
        args = None
        env = None
        process = target.LaunchSimple(
            args, env, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(
            process.GetState() == lldb.eStateStopped,
            PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)

        # Check if global operators are evaluated
        frame = thread.GetSelectedFrame()

        test_result = frame.EvaluateExpression("operator==(s1, s2)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "false",
            "operator==(s1, s2) = false")

        test_result = frame.EvaluateExpression("operator==(s1, s3)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "true",
            "operator==(s1, s3) = true")

        test_result = frame.EvaluateExpression("operator==(s2, s3)")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValue() == "false",
            "operator==(s2, s3) = false")
