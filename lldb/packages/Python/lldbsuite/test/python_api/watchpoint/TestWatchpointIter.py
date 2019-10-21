"""
Use lldb Python SBTarget API to iterate on the watchpoint(s) for the target.
"""

from __future__ import print_function



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointIteratorTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # hardware watchpoints are not reported with a hardware index # on armv7 on ios devices
    def affected_by_radar_34564183(self):
        return (self.getArchitecture() in ['armv7', 'armv7k', 'arm64_32']) and self.platformIsDarwin()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')

    @add_test_categories(['pyapi'])
    def test_watch_iter(self):
        """Exercise SBTarget.watchpoint_iter() API to iterate on the available watchpoints."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint on main.c in order to set our watchpoint later.
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

        # Watch 'global' for read and write.
        value = frame0.FindValue('global', lldb.eValueTypeVariableGlobal)
        error = lldb.SBError()
        watchpoint = value.Watch(True, False, True, error)
        self.assertTrue(value and watchpoint,
                        "Successfully found the variable and set a watchpoint")
        self.DebugSBValue(value)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        # There should be only 1 watchpoint location under the target.
        self.assertTrue(target.GetNumWatchpoints() == 1)
        self.assertTrue(watchpoint.IsEnabled())
        watch_id = watchpoint.GetID()
        self.assertTrue(watch_id != 0)

        # Continue.  Expect the program to stop due to the variable being
        # written to.
        process.Continue()

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        # Print the stack traces.
        lldbutil.print_stacktraces(process)

        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonWatchpoint)
        self.assertTrue(thread, "The thread stopped due to watchpoint")
        self.DebugSBValue(value)

        # We currently only support hardware watchpoint.  Verify that we have a
        # meaningful hardware index at this point.  Exercise the printed repr of
        # SBWatchpointLocation.
        print(watchpoint)
        if not self.affected_by_radar_34564183():
            self.assertTrue(watchpoint.GetHardwareIndex() != -1)

        # SBWatchpoint.GetDescription() takes a description level arg.
        print(lldbutil.get_description(watchpoint, lldb.eDescriptionLevelFull))

        # Now disable the 'rw' watchpoint.  The program won't stop when it reads
        # 'global' next.
        watchpoint.SetEnabled(False)
        self.assertTrue(watchpoint.GetHardwareIndex() == -1)
        self.assertFalse(watchpoint.IsEnabled())

        # Continue.  The program does not stop again when the variable is being
        # read from because the watchpoint location has been disabled.
        process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(
            process.GetState() == lldb.eStateExited,
            PROCESS_EXITED)

        # Verify some vital statistics and exercise the iterator API.
        for watchpoint in target.watchpoint_iter():
            self.assertTrue(watchpoint)
            self.assertTrue(watchpoint.GetWatchSize() == 4)
            self.assertTrue(watchpoint.GetHitCount() == 1)
            print(watchpoint)
