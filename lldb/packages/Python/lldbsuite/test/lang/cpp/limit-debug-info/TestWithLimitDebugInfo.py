import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestWithLimitDebugInfo(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(debug_info=no_match(["dwarf"]))
    def test_limit_debug_info(self):
        self.build()

        cwd = os.getcwd()

        src_file = os.path.join(cwd, "main.cpp")
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "breakpoint file")

        # Get the path of the executable
        exe_path  = os.path.join(cwd, 'a.out')

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        breakpoint = target.BreakpointCreateBySourceRegex("break here", src_file_spec)
        self.assertTrue(breakpoint.IsValid() and breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        thread.StepInto()

        # Get frame for current thread
        frame = thread.GetSelectedFrame()

        v1 = frame.EvaluateExpression("1")
        self.assertTrue(v1.IsValid(), "'expr 1' results in a valid SBValue object")
        self.assertTrue(v1.GetError().Success(), "'expr 1' succeeds without an error.")

        v2 = frame.EvaluateExpression("this")
        self.assertTrue(v2.IsValid(), "'expr this' results in a valid SBValue object")
        self.assertTrue(v2.GetError().Success(), "'expr this' succeeds without an error.")
