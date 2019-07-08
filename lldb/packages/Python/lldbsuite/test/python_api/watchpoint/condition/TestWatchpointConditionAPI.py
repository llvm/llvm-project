"""
Test watchpoint condition API.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointConditionAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source,
                                '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test
        # method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    @expectedFailureAll(
        oslist=["linux"],
        archs=["aarch64"],
        bugnumber="llvm.org/pr27710")
    @skipIfWindows  # Watchpoints not supported on Windows, and this test hangs
    def test_watchpoint_cond_api(self):
        """Test watchpoint condition API."""
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        exe = self.getBuildArtifact(self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Watch 'global' for write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError()
        watchpoint = value.Watch(True, False, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Now set the condition as "global==5".
        watchpoint.SetCondition('global==5')
        self.expect(watchpoint.GetCondition(), exe=False,
                    startstr='global==5')

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print(watchpoint)

        # Continue.  Expect the program to stop due to the variable being
        # written to.
        process.Continue()

        if (self.TraceOn()):
            lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # Verify that the condition is met.
        self.assertTrue(value.GetValueAsUnsigned() == 5)
