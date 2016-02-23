import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCppChainedCalls(TestBase):

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
        exe_path  = os.path.join(cwd, exe_file)

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        main_breakpoint = target.BreakpointCreateBySourceRegex("break here", src_file_spec)
        self.assertTrue(main_breakpoint.IsValid() and main_breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        args = None
        env = None
        process = target.LaunchSimple(args, env, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        # Test chained calls
        test_result = frame.EvaluateExpression("get(set(true))")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "true", "get(set(true)) = true")

        test_result = frame.EvaluateExpression("get(set(false))")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(set(false)) = false")

        test_result = frame.EvaluateExpression("get(t & f)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(t & f) = false")

        test_result = frame.EvaluateExpression("get(f & t)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(f & t) = false")

        test_result = frame.EvaluateExpression("get(t & t)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "true", "get(t & t) = true")

        test_result = frame.EvaluateExpression("get(f & f)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(f & f) = false")

        test_result = frame.EvaluateExpression("get(t & f)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(t & f) = false")

        test_result = frame.EvaluateExpression("get(f) && get(t)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(f) && get(t) = false")

        test_result = frame.EvaluateExpression("get(f) && get(f)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "get(f) && get(t) = false")

        test_result = frame.EvaluateExpression("get(t) && get(t)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "true", "get(t) && get(t) = true")
