import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCppIncompleteTypes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureFreeBSD("llvm.org/pr25626 test executable not built correctly on FreeBSD")
    @skipIfGcc
    def test_limit_debug_info(self):
        self.build()
        frame = self.get_test_frame('limit')

        value_f = frame.EvaluateExpression("f")
        self.assertTrue(value_f.IsValid(), "'expr f' results in a valid SBValue object")
        self.assertTrue(value_f.GetError().Success(), "'expr f' is successful")

        value_a = frame.EvaluateExpression("a")
        self.assertTrue(value_a.IsValid(), "'expr a' results in a valid SBValue object")
        self.assertTrue(value_a.GetError().Success(), "'expr a' is successful")

    @skipIfGcc
    @skipIfWindows # Clang on Windows asserts in external record layout in this case.
    def test_partial_limit_debug_info(self):
        self.build()
        frame = self.get_test_frame('nolimit')

        value_f = frame.EvaluateExpression("f")
        self.assertTrue(value_f.IsValid(), "'expr f' results in a valid SBValue object")
        self.assertTrue(value_f.GetError().Success(), "'expr f' is successful")

        value_a = frame.EvaluateExpression("a")
        self.assertTrue(value_a.IsValid(), "'expr a' results in a valid SBValue object")
        self.assertTrue(value_a.GetError().Success(), "'expr a' is successful")

    def get_test_frame(self, exe):
        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Get the path of the executable
        cwd = os.getcwd()
        exe_path  = os.path.join(cwd, exe)

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
        return thread.GetSelectedFrame()
