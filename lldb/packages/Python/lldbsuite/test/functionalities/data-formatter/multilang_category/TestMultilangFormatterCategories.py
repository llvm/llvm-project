"""
Test that formatter categories can work for multiple languages
"""
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.decorators as decorators
import lldbsuite.test.lldbutil as lldbutil
import os


class TestMultilangFormatterCategories(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipUnlessDarwin
    @decorators.add_test_categories(["swiftpr"])
    def test_multilang_formatter_categories(self):
        """Test that formatter categories can work for multiple languages"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Test that formatter categories can work for multiple languages"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        dic = self.frame.FindVariable("dic")
        lldbutil.check_variable(
            self,
            dic,
            use_dynamic=False,
            summary="2 key/value pairs",
            num_children=2)

        child0 = dic.GetChildAtIndex(0)
        lldbutil.check_variable(
            self,
            child0,
            use_dynamic=False,
            num_children=2,
            typename="__lldb_autogen_nspair")

        id1 = child0.GetChildAtIndex(1)
        lldbutil.check_variable(self, id1, use_dynamic=False, typename="id")

        id1child0 = dic.GetChildAtIndex(1).GetChildAtIndex(0)
        lldbutil.check_variable(
            self,
            id1child0,
            use_dynamic=True,
            typename="NSURL *",
            summary='"http://www.google.com"')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
