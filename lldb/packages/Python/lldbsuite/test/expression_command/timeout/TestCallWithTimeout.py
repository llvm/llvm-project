"""
Test calling a function that waits a while, and make sure the timeout option to expr works.
"""

from __future__ import print_function



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ExprCommandWithTimeoutsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "wait-a-while.cpp"
        self.main_source_spec = lldb.SBFileSpec (self.main_source)


    @expectedFlakeyFreeBSD("llvm.org/pr19605")
    @expectedFlakeyLinux("llvm.org/pr20275")
    @expectedFailureWindows("llvm.org/pr21765")
    def test(self):
        """Test calling std::String member function."""
        self.build()

        exe_name = "a.out"
        exe = os.path.join(os.getcwd(), exe_name)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex('stop here in main.',self.main_source_spec)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)
        self.runCmd("breakpoint list")

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be on self.step_out_of_malloc.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        
        self.assertTrue(len(threads) == 1)
        thread = threads[0]
        
        # First set the timeout too short, and make sure we fail.
        options = lldb.SBExpressionOptions()
        options.SetTimeoutInMicroSeconds(10)
        options.SetUnwindOnError(True)

        frame = thread.GetFrameAtIndex(0)
        
        value = frame.EvaluateExpression ("wait_a_while (50000)", options)
        self.assertTrue (value.IsValid())
        self.assertFalse (value.GetError().Success())

        # Now do the same thing with the command line command, and make sure it works too.
        interp = self.dbg.GetCommandInterpreter()

        result = lldb.SBCommandReturnObject()
        return_value = interp.HandleCommand ("expr -t 100 -u true -- wait_a_while(50000)", result)
        self.assertTrue (return_value == lldb.eReturnStatusFailed)

        # Okay, now do it again with long enough time outs:

        options.SetTimeoutInMicroSeconds(1000000)
        value = frame.EvaluateExpression ("wait_a_while (1000)", options)
        self.assertTrue(value.IsValid())
        self.assertTrue (value.GetError().Success() == True)
        
        # Now do the same thingwith the command line command, and make sure it works too.
        interp = self.dbg.GetCommandInterpreter()

        result = lldb.SBCommandReturnObject()
        return_value = interp.HandleCommand ("expr -t 1000000 -u true -- wait_a_while(1000)", result)
        self.assertTrue(return_value == lldb.eReturnStatusSuccessFinishResult)


        # Finally set the one thread timeout and make sure that doesn't change things much:

        options.SetTimeoutInMicroSeconds(1000000)
        options.SetOneThreadTimeoutInMicroSeconds(500000)
        value = frame.EvaluateExpression ("wait_a_while (1000)", options)
        self.assertTrue(value.IsValid())
        self.assertTrue (value.GetError().Success() == True)
