"""Test that we handle inferiors that send signals to themselves"""

from __future__ import print_function



import os
import lldb
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWindows # signals do not exist on Windows
class RaiseTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_sigstop(self):
        self.build()
        self.signal_test('SIGSTOP', False)
        # passing of SIGSTOP is not correctly handled, so not testing that scenario: https://llvm.org/bugs/show_bug.cgi?id=23574

    @skipIfDarwin # darwin does not support real time signals
    @skipIfTargetAndroid()
    def test_sigsigrtmin(self):
        self.build()
        self.signal_test('SIGRTMIN', True)

    def launch(self, target, signal):
        # launch the process, do not stop at entry point.
        process = target.LaunchSimple([signal], None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a breakpoint")
        return process

    def set_handle(self, signal, pass_signal, stop_at_signal, notify_signal):
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
                "process handle %s -p %s -s %s -n %s" % (signal, pass_signal, stop_at_signal, notify_signal),
                return_obj)
        self.assertTrue (return_obj.Succeeded() == True, "Setting signal handling failed")


    def signal_test(self, signal, test_passing):
        """Test that we handle inferior raising signals"""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        lldbutil.run_break_set_by_symbol(self, "main")

        # launch
        process = self.launch(target, signal)
        signo = process.GetUnixSignals().GetSignalNumberFromName(signal)

        # retrieve default signal disposition
        return_obj = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("process handle %s " % signal, return_obj)
        match = re.match('NAME *PASS *STOP *NOTIFY.*(false|true) *(false|true) *(false|true)',
                return_obj.GetOutput(), re.IGNORECASE | re.DOTALL)
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
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertTrue(thread.GetStopReasonDataCount() >= 1, "There was data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signo,
                "The stop signal was %s" % signal)

        # Continue until we exit.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the notification.
        self.set_handle(signal, "false", "false", "true")
        self.expect("process continue", substrs=["stopped and restarted", signal])
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the notification.
        self.set_handle(signal, "false", "false", "false")
        self.expect("process continue", substrs=["stopped and restarted"], matching=False)
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
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertTrue(thread.GetStopReasonDataCount() >= 1, "There was data in the event.")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0),
                process.GetUnixSignals().GetSignalNumberFromName(signal),
                "The stop signal was %s" % signal)

        # Continue until we exit. The process should receive the signal.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal. We should still get the notification. Process
        # should receive the signal.
        self.set_handle(signal, "true", "false", "true")
        self.expect("process continue", substrs=["stopped and restarted", signal])
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # launch again
        process = self.launch(target, signal)

        # Make sure we do not stop at the signal, and we do not get the notification. Process
        # should receive the signal.
        self.set_handle(signal, "true", "false", "false")
        self.expect("process continue", substrs=["stopped and restarted"], matching=False)
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), signo)

        # reset signal handling to default
        self.set_handle(signal, default_pass, default_stop, default_notify)

    @expectedFailureLinux("llvm.org/pr24530") # the signal the inferior generates gets lost
    @expectedFailureDarwin("llvm.org/pr24530") # the signal the inferior generates gets lost
    def test_restart_bug(self):
        """Test that we catch a signal in the edge case where the process receives it while we are
        about to interrupt it"""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        bkpt = target.BreakpointCreateByName("main")
        self.assertTrue(bkpt.IsValid(), VALID_BREAKPOINT)

        # launch the inferior and don't wait for it to stop
        self.dbg.SetAsync(True)
        error = lldb.SBError()
        listener = lldb.SBListener("my listener")
        process = target.Launch (listener,
                ["SIGSTOP"], # argv
                None,        # envp
                None,        # stdin_path
                None,        # stdout_path
                None,        # stderr_path
                None,        # working directory
                0,           # launch flags
                False,       # Stop at entry
                error)       # error

        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)

        event = lldb.SBEvent()

        # Give the child enough time to reach the breakpoint,
        # while clearing out all the pending events.
        # The last WaitForEvent call will time out after 2 seconds.
        while listener.WaitForEvent(2, event):
            if self.TraceOn():
                print("Process changing state to:", self.dbg.StateAsCString(process.GetStateFromEvent(event)))

        # now the process should be stopped
        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertEqual(len(lldbutil.get_threads_stopped_at_breakpoint(process, bkpt)), 1,
                "A thread should be stopped at breakpoint")

        # Remove all breakpoints. This makes sure we don't have to single-step over them when we
        # resume the process below
        target.DeleteAllBreakpoints()

        # resume the process and immediately try to set another breakpoint. When using the remote
        # stub, this will trigger a request to stop the process just as it is about to stop
        # naturally due to a SIGSTOP signal it raises. Make sure we do not lose this signal.
        process.Continue()
        self.assertTrue(target.BreakpointCreateByName("handler").IsValid(), VALID_BREAKPOINT)

        # Clear the events again
        while listener.WaitForEvent(2, event):
            if self.TraceOn():
                print("Process changing state to:", self.dbg.StateAsCString(process.GetStateFromEvent(event)))

        # The process should be stopped due to a signal
        self.assertEqual(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        self.assertTrue(thread.IsValid(), "Thread should be stopped due to a signal")
        self.assertTrue(thread.GetStopReasonDataCount() >= 1, "There was data in the event.")
        signo = process.GetUnixSignals().GetSignalNumberFromName("SIGSTOP")
        self.assertEqual(thread.GetStopReasonDataAtIndex(0), signo,
                "The stop signal was %s" % signal)

        # We are done
        process.Kill()
