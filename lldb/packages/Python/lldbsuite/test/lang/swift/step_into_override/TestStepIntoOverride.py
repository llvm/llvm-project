import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import platform
import unittest2

class TestStepIntoOverride(lldbtest.TestBase):
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    @skipIfLinux
    def test_swift_stepping(self):
        """Tests that we can step reliably in swift code."""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        # If you are running against a debug swift you are going to
        # end up stepping into the stdlib and that will make stepping
        # tests impossible to write.  So avoid that.

        if platform.system() == 'Darwin':
            lib_name = "libswiftCore.dylib"
        else:
            lib_name = "libswiftCore.so"

        self.dbg.HandleCommand(
            "settings set "
            "target.process.thread.step-avoid-libraries {}".format(lib_name))

    def do_test(self):
        """Tests that we can step reliably in swift code."""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0, lldbtest.VALID_BREAKPOINT)
        
        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        thread = threads[0]

        # Step into the function.
        thread.StepInto()
        frame = thread.frames[0]

        # Make sure we step into the right function, the one that takes
        # an Optional<Int> as argument.
        func = frame.GetFunctionName()
        self.assertEqual(func, 'a.Base.foo(Swift.Optional<Swift.Int>) -> ()')

        # And that we can find the value of `a` after we stepped in.
        valobj = frame.FindVariable('a')
        self.assertEqual(valobj.GetSummary(), '3')
