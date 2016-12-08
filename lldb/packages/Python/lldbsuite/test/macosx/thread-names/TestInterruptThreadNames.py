"""Test that we get thread names when interrupting a process.""" 
from __future__ import print_function


import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInterruptThreadNames(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test that we get thread names when interrupting a process."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        launch_info = lldb.SBLaunchInfo(None)
        error = lldb.SBError()
        lldb.debugger.SetAsync(True)
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        listener = lldb.debugger.GetListener()
        broadcaster = process.GetBroadcaster()
        rc = broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
        self.assertTrue(rc != 0, "Unable to add listener to process")
        self.assertTrue(self.wait_for_running(process, listener), "Check that process is up and running")


        inferior_set_up = lldb.SBValue()
        retry = 5
        while retry > 0:
            time.sleep(1)
            process.SendAsyncInterrupt()
            self.assertTrue(self.wait_for_stop(process, listener), "Check that process is paused")
            inferior_set_up = target.CreateValueFromExpression("threads_up_and_running", "threads_up_and_running")
            if inferior_set_up.IsValid() and inferior_set_up.GetValueAsSigned() == 1:
                retry = 0
            else:
                process.Continue()
            retry = retry - 1

        self.assertTrue(inferior_set_up.IsValid() and inferior_set_up.GetValueAsSigned() == 1, "Check that the program was able to create its threads within the allotted time")

        self.check_number_of_threads(process)

        main_thread = lldb.SBThread()
        second_thread = lldb.SBThread()
        third_thread = lldb.SBThread()
        for idx in range(0, process.GetNumThreads()):
            t = process.GetThreadAtIndex(idx)
            if t.GetName() == "main thread":
                main_thread = t
            if t.GetName() == "second thread":
                second_thread = t
            if t.GetName() == "third thread":
                third_thread = t

        self.assertTrue(
            main_thread.IsValid() and second_thread.IsValid() and third_thread.IsValid(),
            "Got all three expected threads")

        process.Kill()

    def wait_for_running(self, process, listener):
        retry_count = 5
        if process.GetState() == lldb.eStateRunning:
            return True

        while retry_count > 0:
            event = lldb.SBEvent()
            listener.WaitForEvent(2, event)
            if event.GetType() == lldb.SBProcess.eBroadcastBitStateChanged:
                if process.GetState() == lldb.eStateRunning:
                    return True
            retry_count = retry_count - 1

        return False

    def check_number_of_threads(self, process):
        self.assertTrue(
            process.GetNumThreads() == 3,
            "Check that the process has three threads when sitting at the stopper() breakpoint")



    def wait_for_stop(self, process, listener):
        retry_count = 5
        if process.GetState() == lldb.eStateStopped or process.GetState() == lldb.eStateCrashed or process.GetState() == lldb.eStateDetached or process.GetState() == lldb.eStateExited:
            return True

        while retry_count > 0:
            event = lldb.SBEvent()
            listener.WaitForEvent(2, event)
            if event.GetType() == lldb.SBProcess.eBroadcastBitStateChanged:
                if process.GetState() == lldb.eStateStopped or process.GetState() == lldb.eStateCrashed or process.GetState() == lldb.eStateDetached or process.GetState() == lldb.eStateExited:
                    return True
            retry_count = retry_count - 1

        return False


