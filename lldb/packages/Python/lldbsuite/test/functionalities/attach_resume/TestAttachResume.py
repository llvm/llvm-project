"""
Test process attach/resume.
"""

from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

exe_name = "AttachResume"  # Must match Makefile


class AttachResumeTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr19310')
    @expectedFailureNetBSD
    @skipIfWindows # llvm.org/pr24778, llvm.org/pr21753
    def test_attach_continue_interrupt_detach(self):
        """Test attach/continue/interrupt/detach"""
        self.build()
        self.process_attach_continue_interrupt_detach()

    def process_attach_continue_interrupt_detach(self):
        """Test attach/continue/interrupt/detach"""

        exe = self.getBuildArtifact(exe_name)

        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -p " + str(popen.pid))

        self.setAsync(True)
        listener = self.dbg.GetListener()
        process = self.dbg.GetSelectedTarget().GetProcess()

        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateRunning])

        self.runCmd("process interrupt")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateStopped])

        # be sure to continue/interrupt/continue (r204504)
        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateRunning])

        self.runCmd("process interrupt")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateStopped])

        # Second interrupt should have no effect.
        self.expect(
            "process interrupt",
            patterns=["Process is not running"],
            error=True)

        # check that this breakpoint is auto-cleared on detach (r204752)
        self.runCmd("br set -f main.cpp -l %u" %
                    (line_number('main.cpp', '// Set breakpoint here')))

        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateRunning, lldb.eStateStopped])
        self.expect('br list', 'Breakpoint not hit',
                    substrs=['hit count = 1'])

        # Make sure the breakpoint is not hit again.
        self.expect("expr debugger_flag = false", substrs=[" = false"])

        self.runCmd("c")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateRunning])

        # make sure to detach while in running state (r204759)
        self.runCmd("detach")
        lldbutil.expect_state_changes(
            self, listener, process, [
                lldb.eStateDetached])
