"""
Test Multi thread.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_threads


class MultiThreadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_multi_thread(self):
        """Use Python APIs to check multi-thread."""
        self.build()
        self.do_test_multi_thread()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.bkp_line = line_number('main.c', '// Breakpoint location')
        self.step_1_line = line_number('main.c', '// Step location 1')
        self.step_2_line = line_number('main.c', '// Step location 2')

    def thread_is_stopped(self, threads, ref_thread):
        for thread in threads:
            self.assertTrue(thread.IsValid())
            if thread.GetThreadID() == ref_thread.GetThreadID():
                return True
        return False

    def do_test_multi_thread(self):
        exe = self.getBuildArtifact("a.out")
        filespec = lldb.SBFileSpec("main.c", False)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(filespec, self.bkp_line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        thread = process.GetThreadAtIndex(8)
        threads = get_stopped_threads(process, lldb.eStopReasonBreakpoint)
        while not self.thread_is_stopped(threads, thread):
            process.Continue()
            threads = get_stopped_threads(process, lldb.eStopReasonBreakpoint)

        target.BreakpointDelete(breakpoint.GetID())

        frame0 = thread.GetFrameAtIndex(0)
        while frame0.GetLineEntry().GetLine() != self.step_1_line:
            thread.StepOver()
            self.assertTrue(thread.GetStopReason()
                            == lldb.eStopReasonPlanComplete)
            self.assertTrue(thread.GetNumFrames() == 2)
            frame0 = thread.GetFrameAtIndex(0)

        thread.StepInto()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.GetNumFrames() == 3)

        frame0 = thread.GetFrameAtIndex(0)
        while frame0.GetLineEntry().GetLine() != self.step_2_line:
            thread.StepOver()
            self.assertTrue(thread.GetStopReason()
                            == lldb.eStopReasonPlanComplete)
            self.assertTrue(thread.GetNumFrames() == 3)
            frame0 = thread.GetFrameAtIndex(0)

        frame0 = thread.GetFrameAtIndex(0)
        self.runCmd("disassemble --pc --count 1")
        while not re.search("call", self.res.GetOutput()):
            thread.StepInstruction(False)
            self.assertTrue(thread.GetStopReason()
                            == lldb.eStopReasonPlanComplete)
            self.assertTrue(thread.GetNumFrames() == 3)
            self.runCmd("disassemble --pc --count 1")

        thread.StepInstruction(False)
        self.assertTrue(thread.GetStopReason()
                        == lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.GetNumFrames() == 4)

        frame0 = thread.GetFrameAtIndex(0)
        while frame0.GetLineEntry().GetLine() != self.step_2_line:
            thread.StepOver()
            self.assertTrue(thread.GetStopReason()
                            == lldb.eStopReasonPlanComplete)
            frame0 = thread.GetFrameAtIndex(0)

        self.runCmd("disassemble --pc --count 1")
        while not re.search("call", self.res.GetOutput()):
            thread.StepInstruction(False)
            self.assertTrue(thread.GetStopReason()
                            == lldb.eStopReasonPlanComplete)
            self.assertTrue(thread.GetNumFrames() == 4)
            self.runCmd("disassemble --pc --count 1")

        thread.StepInstruction(True)
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.GetNumFrames() == 4)

        thread.StepOut()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.GetNumFrames() == 3)

        thread.StepOut()
        self.assertTrue(thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        self.assertTrue(thread.GetNumFrames() == 2)

        process.Continue()
        self.assertTrue(process.GetState() == lldb.eStateExited)
        self.assertTrue(process.GetExitStatus() == 0x9)
