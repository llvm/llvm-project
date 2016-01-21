"""
Test continue from a breakpoint when there is a breakpoint on the next instruction also.
"""

from __future__ import print_function



import unittest2
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ConsecutiveBreakpoitsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll("llvm.org/pr23478", oslist = not_in(["macosx"]))
    def test (self):
        self.build ()
        self.consecutive_breakpoints_tests()
        
    def consecutive_breakpoints_tests(self):
        exe = os.path.join (os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint1 = target.BreakpointCreateBySourceRegex("Set breakpoint here", lldb.SBFileSpec("main.cpp"))
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should be stopped at the first breakpoint
        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, breakpoint1)
        self.assertIsNotNone(thread, "Expected one thread to be stopped at breakpoint 1")

        # Set breakpoint to the next instruction
        frame = thread.GetFrameAtIndex(0)
        
        address = frame.GetPCAddress()
        instructions = target.ReadInstructions(address, 2)
        self.assertTrue(len(instructions) == 2)
        address = instructions[1].GetAddress()
        
        breakpoint2 = target.BreakpointCreateByAddress(address.GetLoadAddress(target))
        process.Continue()

        # We should be stopped at the second breakpoint
        thread = lldbutil.get_one_thread_stopped_at_breakpoint(process, breakpoint2)
        self.assertIsNotNone(thread, "Expected one thread to be stopped at breakpoint 2")

        # Run the process until termination
        process.Continue()
