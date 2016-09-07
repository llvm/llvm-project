"""
Test scopes in C++.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppScopes(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureDarwin
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
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

        # Get current fream of the thread at the breakpoint
        frame = thread.GetSelectedFrame()

        # Test result for scopes of variables

        global_variables = frame.GetVariables(True, True, True, False)
        global_variables_assert = {
            'A::a': 1111,
            'B::a': 2222,
            'C::a': 3333,
            '::a': 4444,
            'a': 4444
        }

        self.assertTrue(
            global_variables.GetSize() == 4,
            "target variable returns all variables")
        for variable in global_variables:
            name = variable.GetName()
            self.assertTrue(
                name in global_variables_assert,
                "target variable returns wrong variable " + name)

        for name in global_variables_assert:
            value = frame.EvaluateExpression(name)
            assert_value = global_variables_assert[name]
            self.assertTrue(
                value.IsValid() and value.GetValueAsSigned() == assert_value,
                name + " = " + str(assert_value))
