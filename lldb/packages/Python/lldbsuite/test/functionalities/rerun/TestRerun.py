"""
Test that argdumper is a viable launching strategy.
"""
from __future__ import print_function



import lldb
import os
import time
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class TestRerun(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test (self):
        self.build()
        exe = os.path.join (os.getcwd(), "a.out")
        
        self.runCmd("target create %s" % exe)
        
        # Create the target
        target = self.dbg.CreateTarget(exe)
        
        # Create any breakpoints we need
        breakpoint = target.BreakpointCreateBySourceRegex ('break here', lldb.SBFileSpec ("main.cpp", False))
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        self.runCmd("process launch 1 2 3")

        process = self.process()
        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, breakpoint)
        self.assertIsNotNone(thread, "Process should be stopped at a breakpoint in main")
        self.assertTrue(thread.IsValid(), "Stopped thread is not valid")

        self.expect("frame variable argv[1]", substrs=['1'])
        self.expect("frame variable argv[2]", substrs=['2'])
        self.expect("frame variable argv[3]", substrs=['3'])
        
        # Let program exit
        self.runCmd("continue")
        
        # Re-run with no args and make sure we still run with 1 2 3 as arguments as
        # they should have been stored in "target.run-args"
        self.runCmd("process launch")

        process = self.process()
        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, breakpoint)

        self.assertIsNotNone(thread, "Process should be stopped at a breakpoint in main");
        self.assertTrue(thread.IsValid(), "Stopped thread is not valid")

        self.expect("frame variable argv[1]", substrs=['1'])
        self.expect("frame variable argv[2]", substrs=['2'])
        self.expect("frame variable argv[3]", substrs=['3'])
