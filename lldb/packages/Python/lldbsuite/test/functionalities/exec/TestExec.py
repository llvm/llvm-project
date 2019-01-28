"""
Test some lldb command abbreviations.
"""
from __future__ import print_function


import lldb
import os
import time
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExecTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @expectedFailureAll(archs=['i386'], bugnumber="rdar://28656532")
    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://problem/34559552") # this exec test has problems on ios systems
    @skipIfSanitized # rdar://problem/43756823
    def test_hitting_exec (self):
        self.do_test(False)

    @skipUnlessDarwin
    @expectedFailureAll(archs=['i386'], bugnumber="rdar://28656532")
    @expectedFailureAll(oslist=["ios", "tvos", "watchos", "bridgeos"], bugnumber="rdar://problem/34559552") # this exec test has problems on ios systems
    @skipIfSanitized # rdar://problem/43756823
    def test_skipping_exec (self):
        self.do_test(False)

    def do_test(self, skip_exec):
        self.build()
        exe = self.getBuildArtifact("a.out")
        secondprog = self.getBuildArtifact("secondprog")

        # Create the target
        target = self.dbg.CreateTarget(exe)

        # Create any breakpoints we need
        breakpoint1 = target.BreakpointCreateBySourceRegex(
            'Set breakpoint 1 here', lldb.SBFileSpec("main.cpp", False))
        self.assertTrue(breakpoint1, VALID_BREAKPOINT)
        breakpoint2 = target.BreakpointCreateBySourceRegex(
            'Set breakpoint 2 here', lldb.SBFileSpec("secondprog.cpp", False))
        self.assertTrue(breakpoint2, VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        if self.TraceOn():
            self.runCmd("settings show target.process.stop-on-exec", check=False)
        if skip_exec:
            self.dbg.HandleCommand("settings set target.process.stop-on-exec false")
            def cleanup():
                self.runCmd("settings set target.process.stop-on-exec false",
                            check=False)

            # Execute the cleanup function during test case tear down.
            self.addTearDownHook(cleanup)

        # The stop reason of the thread should be breakpoint.
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
        process, breakpoint1)
        self.assertTrue(len(threads) == 1)

        # We had a deadlock tearing down the TypeSystemMap on exec, but only if some
        # expression had been evaluated.  So make sure we do that here so the teardown
        # is not trivial.

        thread = threads[0]
        value = thread.frames[0].EvaluateExpression("1 + 2")
        self.assertTrue(
            value.IsValid(),
            "Expression evaluated successfully")
        int_value = value.GetValueAsSigned()
        self.assertTrue(int_value == 3, "Expression got the right result.")

        # Run and we should stop due to exec
        process.Continue()

        if not skip_exec:
            self.assertFalse(process.GetState() == lldb.eStateExited,
                             "Process should not have exited!")
            self.assertTrue(process.GetState() == lldb.eStateStopped,
                            "Process should be stopped at __dyld_start")

            threads = lldbutil.get_stopped_threads(
                process, lldb.eStopReasonExec)
            self.assertTrue(
                len(threads) == 1,
                "We got a thread stopped for exec.")

            # Run and we should stop at breakpoint in main after exec
            process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint2)
        if self.TraceOn():
            for t in process.threads:
                print(t)
                if t.GetStopReason() != lldb.eStopReasonBreakpoint:
                    self.runCmd("bt")
        self.assertTrue(len(threads) == 1,
                        "Stopped at breakpoint in exec'ed process.")
