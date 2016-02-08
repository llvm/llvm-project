"""
Use lldb Python SBWatchpoint API to set the ignore count.
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class WatchpointIgnoreCountTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')

    @add_test_categories(['pyapi'])
    @expectedFailureAndroid(archs=['arm', 'aarch64']) # Watchpoints not supported
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24446: WINDOWS XFAIL TRIAGE - Watchpoints not supported on Windows")
    def test_set_watch_ignore_count(self):
        """Test SBWatchpoint.SetIgnoreCount() API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint on main.c in order to set our watchpoint later.
        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        # We should be stopped due to the breakpoint.  Get frame #0.
        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Watch 'global' for read and write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError();
        watchpoint = value.Watch(True, True, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        # There should be only 1 watchpoint location under the target.
        self.assertTrue(target.GetNumWatchpoints() == 1)
        watchpoint = target.GetWatchpointAtIndex(0)
        self.assertTrue(watchpoint.IsEnabled())
        self.assertTrue(watchpoint.GetIgnoreCount() == 0)
        watch_id = watchpoint.GetID()
        self.assertTrue(watch_id != 0)
        print(watchpoint)

        # Now immediately set the ignore count to 2.  When we continue, expect the
        # inferior to run to its completion without stopping due to watchpoint.
        watchpoint.SetIgnoreCount(2)
        print(watchpoint)
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

        # Verify some vital statistics.
        self.assertTrue(watchpoint)
        self.assertTrue(watchpoint.GetWatchSize() == 4)
        self.assertTrue(watchpoint.GetHitCount() == 2)
        print(watchpoint)
