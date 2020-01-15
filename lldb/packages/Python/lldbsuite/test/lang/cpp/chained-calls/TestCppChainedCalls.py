import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppChainedCalls(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        self.build()

        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        main_breakpoint = target.BreakpointCreateBySourceRegex(
            "break here", src_file_spec)
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
        lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Test chained calls
        self.expect_expr("get(set(true))", result_type="bool", result_value="true")
        self.expect_expr("get(set(false))", result_type="bool", result_value="false")
        self.expect_expr("get(t & f)", result_type="bool", result_value="false")
        self.expect_expr("get(f & t)", result_type="bool", result_value="false")
        self.expect_expr("get(t & t)", result_type="bool", result_value="true")
        self.expect_expr("get(f & f)", result_type="bool", result_value="false")
        self.expect_expr("get(t & f)", result_type="bool", result_value="false")
        self.expect_expr("get(f) && get(t)", result_type="bool", result_value="false")
        self.expect_expr("get(f) && get(f)", result_type="bool", result_value="false")
        self.expect_expr("get(t) && get(t)", result_type="bool", result_value="true")
