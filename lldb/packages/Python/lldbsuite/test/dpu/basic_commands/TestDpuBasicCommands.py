"""
Test basic commands.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import get_stopped_thread


class BasicCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(['pyapi'])
    def test_basic_commands(self):
        """Use Python APIs to check basic commands."""
        self.build()
        self.do_test_basic_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_first_line = line_number('main.c', '// Breakpoint location 1')
        self.fct1_call_line = line_number('main.c', '// Breakpoint location 2')
        self.step_line = line_number('main.c', ' // Step location')
        self.step_in_entry_line = line_number('main.c', '// StepIn entry location')

    def do_test_basic_commands(self):
        exe = self.getBuildArtifact("a.out")
        filespec = lldb.SBFileSpec("main.c", False)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation(filespec, self.main_first_line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        breakpoint = target.BreakpointCreateByLocation(filespec, self.fct1_call_line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(
            frame0.GetLineEntry().GetLine() == self.main_first_line,
            "Thread did not stop at first line of main function")

        process.Continue()

        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(
            frame0.GetLineEntry().GetLine() == self.fct1_call_line,
            "Thread did not stop at the call of function fct1")
        while not re.search("->.*: call", frame0.Disassemble()) :
            thread.StepInstruction(False)
        self.assertTrue(thread.GetNumFrames() == 2)

        thread.StepInstruction(False)
        self.assertTrue(thread.GetNumFrames() == 3)
        frame0 = thread.GetFrameAtIndex(0)
        while frame0.GetLineEntry().GetLine() != self.step_line :
            thread.StepOver()
            frame0 = thread.GetFrameAtIndex(0)
            self.assertTrue(thread.GetNumFrames() == 3)

        thread.StepInto()
        self.assertTrue(thread.GetNumFrames() == 4)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.step_in_entry_line)

        while thread.GetNumFrames() == 4 :
            thread.StepInstruction(True)

        self.assertTrue(thread.GetNumFrames() == 3)

        thread.StepOut()
        self.assertTrue(thread.GetNumFrames() == 2)
        thread.StepOut()
        self.assertTrue(thread.GetNumFrames() == 1)

        process.Continue()

        self.assertTrue(process.GetState() == lldb.eStateExited)
        self.assertTrue(process.GetExitStatus() == 0x63)
