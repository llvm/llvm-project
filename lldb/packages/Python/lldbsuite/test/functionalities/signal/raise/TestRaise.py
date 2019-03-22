"""Test that we handle inferiors that send signals to themselves"""

from __future__ import print_function


import os
import lldb
import re
from lldbsuite.test.lldbplatformutil import getDarwinOSTriples
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWindows  # signals do not exist on Windows
class RaiseTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNetBSD  # Hangs on NetBSD
    def test_sigstop(self):
        self.build()
        self.signal_test('SIGSTOP', False)
        # passing of SIGSTOP is not correctly handled, so not testing that
        # scenario: https://llvm.org/bugs/show_bug.cgi?id=23574

    @skipIfDarwin  # darwin does not support real time signals
    @skipIfTargetAndroid()
    def test_sigsigrtmin(self):
        self.build()
        self.signal_test('SIGRTMIN', True)

    @skipIfNetBSD  # Hangs on NetBSD
    def test_sigtrap(self):
        self.build()
        self.signal_test('SIGTRAP', True)

    def launch(self, target, signal):
        # launch the process, do not stop at entry point.
        process = target.LaunchSimple(
            [signal], None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "Thread should be stopped due to a breakpoint")
        return process

    def set_handle(self, signal, pass_signal, stop_at_signal, notify_signal):
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle %s -p %s -s %s -n %s" %
            (signal, pass_signal, stop_at_signal, notify_signal), return_obj)
        self.assertTrue(
            return_obj.Succeeded(),
            "Setting signal handling failed")

    def signal_test(self, signal, test_passing):
        """Test that we handle inferior raising signals"""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        lldbutil.run_break_set_by_symbol(self, "main")

        # launch
        process = self.launch(target, signal)
        signo = process.GetUnixSignals().GetSignalNumberFromName(signal)

        # retrieve default signal disposition
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "process handle %s " % signal, return_obj)
        match = re.match(
            'NAME *PASS *STOP *NOTIFY.*(false|true) *(false|true) *(false|true)',
            return_obj.GetOutput(),
            re.IGNORECASE | re.DOTALL)
        if not match:
            self.fail('Unable to retrieve default signal disposition.')
        default_pass = match.group(1)
        default_stop = match.group(2)
        default_notify = match.group(3)

        # Make sure we stop at the signal
        self.set_handle(signal, "false", "true", "true")
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(
            thread.IsValid(),
            "Thread should be stopped due to a signal")
        self.assertTrue(
            thread.GetStopReasonDataCount() >= 1,
            "There was data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signo,
                         "The stop signal was %s" % signal)

        # Continue until we exit.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the
        # notification.
        self.set_handle(signal, "false", "false", "true")
        self.expect(
            "process continue",
            substrs=[
                "stopped and restarted",
                signal])
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the
        # notification.
        self.set_handle(signal, "false", "false", "false")
        self.expect(
            "process continue",
            substrs=["stopped and restarted"],
            matching=False)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        if not test_passing:
            # reset signal handling to default
            self.set_handle(signal, default_pass, default_stop, default_notify)
            return

        # launch again
        process = self.launch(target, signal)

        # Make sure we stop at the signal
        self.set_handle(signal, "true", "true", "true")
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(
            thread.IsValid(),
            "Thread should be stopped due to a signal")
        self.assertTrue(
            thread.GetStopReasonDataCount() >= 1,
            "There was data in the event.")
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            process.GetUnixSignals().GetSignalNumberFromName(signal),
            "The stop signal was %s" %
            signal)

        # Continue until we exit. The process should receive the signal.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the notification. Process
        # should receive the signal.
        self.set_handle(signal, "true", "false", "true")
        self.expect(
            "process continue",
            substrs=[
                "stopped and restarted",
                signal])
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the notification. Process
        # should receive the signal.
        self.set_handle(signal, "true", "false", "false")
        self.expect(
            "process continue",
            substrs=["stopped and restarted"],
            matching=False)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # reset signal handling to default
        self.set_handle(signal, default_pass, default_stop, default_notify)
