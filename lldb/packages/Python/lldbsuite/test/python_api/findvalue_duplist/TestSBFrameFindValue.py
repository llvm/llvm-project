"""Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBFrameFindValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_formatters_api(self):
        """Test that SBFrame::FindValue finds things but does not duplicate the entire variables list"""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', lldb.SBFileSpec("main.cpp"))
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
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        self.assertTrue(
            self.frame.GetVariables(
                True,
                True,
                False,
                True).GetSize() == 2,
            "variable count is off")
        self.assertFalse(
            self.frame.FindValue(
                "NoSuchThing",
                lldb.eValueTypeVariableArgument,
                lldb.eDynamicCanRunTarget).IsValid(),
            "found something that should not be here")
        self.assertTrue(
            self.frame.GetVariables(
                True,
                True,
                False,
                True).GetSize() == 2,
            "variable count is off after failed FindValue()")
        self.assertTrue(
            self.frame.FindValue(
                "a",
                lldb.eValueTypeVariableArgument,
                lldb.eDynamicCanRunTarget).IsValid(),
            "FindValue() didn't find an argument")
        self.assertTrue(
            self.frame.GetVariables(
                True,
                True,
                False,
                True).GetSize() == 2,
            "variable count is off after successful FindValue()")
