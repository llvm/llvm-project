"""
Test that argdumper is a viable launching strategy.
"""
from __future__ import print_function



import lldb
import os
import time
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LaunchWithShellExpandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureFreeBSD("llvm.org/pr22627 process launch w/ shell expansion not working")
    @expectedFailureLinux("llvm.org/pr22627 process launch w/ shell expansion not working")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test(self):
        self.build()
        exe = os.path.join (os.getcwd(), "a.out")
        
        self.runCmd("target create %s" % exe)
        
        # Create the target
        target = self.dbg.CreateTarget(exe)
        
        # Create any breakpoints we need
        breakpoint = target.BreakpointCreateBySourceRegex ('break here', lldb.SBFileSpec ("main.cpp", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        self.runCmd("process launch -X true -w %s -- fi*.tx?" % (os.getcwd()))

        process = self.process()

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex (0)

        self.assertTrue (thread.IsValid(),
                         "Process stopped at 'main' should have a valid thread");

        stop_reason = thread.GetStopReason()
        
        self.assertTrue (stop_reason == lldb.eStopReasonBreakpoint,
                         "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint");

        self.expect("frame variable argv[1]", substrs=['file1.txt'])
        self.expect("frame variable argv[2]", substrs=['file2.txt'])
        self.expect("frame variable argv[3]", substrs=['file3.txt'])
        self.expect("frame variable argv[4]", substrs=['file4.txy'])
        self.expect("frame variable argv[5]", substrs=['file5.tyx'], matching=False)

        self.runCmd("process kill")

        self.runCmd('process launch -X true -w %s -- "foo bar"' % (os.getcwd()))
        
        process = self.process()

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex (0)

        self.assertTrue (thread.IsValid(),
                         "Process stopped at 'main' should have a valid thread");

        stop_reason = thread.GetStopReason()
        
        self.assertTrue (stop_reason == lldb.eStopReasonBreakpoint,
                         "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint");
        
        self.expect("frame variable argv[1]", substrs=['foo bar'])

        self.runCmd("process kill")

        self.runCmd('process launch -X true -w %s -- foo\ bar' % (os.getcwd()))
        
        process = self.process()

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        thread = process.GetThreadAtIndex (0)

        self.assertTrue (thread.IsValid(),
                         "Process stopped at 'main' should have a valid thread");

        stop_reason = thread.GetStopReason()
        
        self.assertTrue (stop_reason == lldb.eStopReasonBreakpoint,
                         "Thread in process stopped in 'main' should have a stop reason of eStopReasonBreakpoint");
        
        self.expect("frame variable argv[1]", substrs=['foo bar'])
