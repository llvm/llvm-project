"""
Test that global operators are found and evaluated.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppGlobalOperators(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def prepare_executable_and_get_frame(self):
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

        return thread.GetSelectedFrame()

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_equals_operator(self):
        frame = self.prepare_executable_and_get_frame()

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

    def do_new_test(self, frame, expr, expected_value_name):
        """Evaluate a new expression, and check its result"""

        expected_value = frame.FindValue(
            expected_value_name, lldb.eValueTypeVariableGlobal)
        self.assertTrue(expected_value.IsValid())

        expected_value_addr = expected_value.AddressOf()
        self.assertTrue(expected_value_addr.IsValid())

        got = frame.EvaluateExpression(expr)
        self.assertTrue(got.IsValid())
        self.assertEqual(
            got.GetValueAsUnsigned(),
            expected_value_addr.GetValueAsUnsigned())
        got_type = got.GetType()
        self.assertTrue(got_type.IsPointerType())
        self.assertEqual(got_type.GetPointeeType().GetName(), "Struct")

    def test_operator_new(self):
        frame = self.prepare_executable_and_get_frame()

        self.do_new_test(frame, "new Struct()", "global_new_buf")
        self.do_new_test(frame, "new(new_tag) Struct()", "tagged_new_buf")
