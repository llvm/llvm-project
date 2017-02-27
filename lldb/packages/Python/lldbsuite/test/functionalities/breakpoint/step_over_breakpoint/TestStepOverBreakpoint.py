"""
Test that breakpoints do not affect stepping.
Check for correct StopReason when stepping to the line with breakpoint 
which chould be eStopReasonBreakpoint in general,
and eStopReasonPlanComplete when breakpoint's condition fails.  
"""

from __future__ import print_function

import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class StepOverBreakpointsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
       
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")
        src = lldb.SBFileSpec("main.cpp")

        # Create a target by the debugger.
        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        # Setup four breakpoints, two of them with false condition
        self.line1 = line_number('main.cpp', "breakpoint_1")
        self.line4 = line_number('main.cpp', "breakpoint_4")

        self.breakpoint1 = self.target.BreakpointCreateByLocation(src, self.line1)        
        self.assertTrue(
            self.breakpoint1 and self.breakpoint1.GetNumLocations() == 1,
            VALID_BREAKPOINT)

        self.breakpoint2 = self.target.BreakpointCreateBySourceRegex("breakpoint_2", src)
        self.breakpoint2.GetLocationAtIndex(0).SetCondition('false')

        self.breakpoint3 = self.target.BreakpointCreateBySourceRegex("breakpoint_3", src)
        self.breakpoint3.GetLocationAtIndex(0).SetCondition('false')

        self.breakpoint4 = self.target.BreakpointCreateByLocation(src, self.line4)

        # Start debugging
        self.process = self.target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertIsNotNone(self.process, PROCESS_IS_VALID)
        self.thread = lldbutil.get_one_thread_stopped_at_breakpoint(self.process, self.breakpoint1)
        self.assertIsNotNone(self.thread, "Didn't stop at breakpoint 1.")

    def test_step_instruction(self): 
        # Count instructions between breakpoint_1 and breakpoint_4
        contextList = self.target.FindFunctions('main', lldb.eFunctionNameTypeAuto)
        self.assertEquals(contextList.GetSize(), 1)
        symbolContext = contextList.GetContextAtIndex(0)
        function = symbolContext.GetFunction()
        self.assertTrue(function)
        instructions = function.GetInstructions(self.target)
        addr_1 = self.breakpoint1.GetLocationAtIndex(0).GetAddress()
        addr_4 = self.breakpoint4.GetLocationAtIndex(0).GetAddress()
        for i in range(instructions.GetSize()) :
            addr = instructions.GetInstructionAtIndex(i).GetAddress()
            if (addr == addr_1) : index_1 = i
            if (addr == addr_4) : index_4 = i 

        steps_expected = index_4 - index_1
        step_count = 0
        # Step from breakpoint_1 to breakpoint_4
        while True:
            self.thread.StepInstruction(True)
            step_count = step_count + 1
            self.assertEquals(self.process.GetState(), lldb.eStateStopped)
            self.assertTrue(self.thread.GetStopReason() == lldb.eStopReasonPlanComplete or
                            self.thread.GetStopReason() == lldb.eStopReasonBreakpoint)
            if (self.thread.GetStopReason() == lldb.eStopReasonBreakpoint) :
                # we should not stop on breakpoint_2 and _3 because they have false condition
                self.assertEquals(self.thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.line4)
                # breakpoint_2 and _3 should not affect step count
                self.assertTrue(step_count >= steps_expected)
                break

        # Run the process until termination
        self.process.Continue()
        self.assertEquals(self.process.GetState(), lldb.eStateExited)

    @skipIf(bugnumber="llvm.org/pr31972", hostoslist=["windows"])
    def test_step_over(self):
        #lldb.DBG.EnableLog("lldb", ["step","breakpoint"])
        
        self.thread.StepOver()
        # We should be stopped at the breakpoint_2 line with stop plan complete reason
        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(self.thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        self.thread.StepOver()
        # We should be stopped at the breakpoint_3 line with stop plan complete reason
        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(self.thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        self.thread.StepOver()
        # We should be stopped at the breakpoint_4
        self.assertEquals(self.process.GetState(), lldb.eStateStopped)
        self.assertEquals(self.thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        thread1 = lldbutil.get_one_thread_stopped_at_breakpoint(self.process, self.breakpoint4)
        self.assertEquals(self.thread, thread1, "Didn't stop at breakpoint 4.")

        # Check that stepping does not affect breakpoint's hit count
        self.assertEquals(self.breakpoint1.GetHitCount(), 1)
        self.assertEquals(self.breakpoint2.GetHitCount(), 0)
        self.assertEquals(self.breakpoint3.GetHitCount(), 0)
        self.assertEquals(self.breakpoint4.GetHitCount(), 1)

        # Run the process until termination
        self.process.Continue()
        self.assertEquals(self.process.GetState(), lldb.eStateExited)

