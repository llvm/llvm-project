"""Test queues inspection SB APIs."""

from __future__ import print_function



import unittest2
import os, time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestQueues(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])      
    def test_with_python_api(self):
        """Test queues inspection SB APIs."""
        self.build()
        self.queues()
        self.queues_with_libBacktraceRecording()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    def check_queue_for_valid_queue_id(self, queue):
        self.assertTrue(queue.GetQueueID() != 0, "Check queue %s for valid QueueID (got 0x%x)" % (queue.GetName(), queue.GetQueueID()))

    def check_running_and_pending_items_on_queue(self, queue, expected_running, expected_pending):
        self.assertTrue(queue.GetNumPendingItems() == expected_pending, "queue %s should have %d pending items, instead has %d pending items" % (queue.GetName(), expected_pending, (queue.GetNumPendingItems())))
        self.assertTrue(queue.GetNumRunningItems() == expected_running, "queue %s should have %d running items, instead has %d running items" % (queue.GetName(), expected_running, (queue.GetNumRunningItems())))

    def check_number_of_threads_owned_by_queue(self, queue, number_threads):
        self.assertTrue(queue.GetNumThreads() == number_threads, "queue %s should have %d thread executing, but has %d" % (queue.GetName(), number_threads, queue.GetNumThreads()))

    def check_queue_kind (self, queue, kind):
        expected_kind_string = "Unknown"
        if kind == lldb.eQueueKindSerial:
            expected_kind_string = "Serial queue"
        if kind == lldb.eQueueKindConcurrent:
            expected_kind_string = "Concurrent queue"
        actual_kind_string = "Unknown"
        if queue.GetKind() == lldb.eQueueKindSerial:
            actual_kind_string = "Serial queue"
        if queue.GetKind() == lldb.eQueueKindConcurrent:
            actual_kind_string = "Concurrent queue"
        self.assertTrue(queue.GetKind() == kind, "queue %s is expected to be a %s but it is actually a %s" % (queue.GetName(), expected_kind_string, actual_kind_string))

    def check_queues_threads_match_queue(self, queue):
        for idx in range(0, queue.GetNumThreads()):
            t = queue.GetThreadAtIndex(idx)
            self.assertTrue(t.IsValid(), "Queue %s's thread #%d must be valid" % (queue.GetName(), idx))
            self.assertTrue(t.GetQueueID() == queue.GetQueueID(), "Queue %s has a QueueID of %d but its thread #%d has a QueueID of %d" % (queue.GetName(), queue.GetQueueID(), idx, t.GetQueueID()))
            self.assertTrue(t.GetQueueName() == queue.GetName(), "Queue %s has a QueueName of %s but its thread #%d has a QueueName of %s" % (queue.GetName(), queue.GetName(), idx, t.GetQueueName()))
            self.assertTrue(t.GetQueue().GetQueueID() == queue.GetQueueID(), "Thread #%d's Queue's QueueID of %d is not the same as the QueueID of its owning queue %d" % (idx, t.GetQueue().GetQueueID(), queue.GetQueueID()))

    def queues(self):
        """Test queues inspection SB APIs without libBacktraceRecording."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.main_source_spec = lldb.SBFileSpec (self.main_source)
        break1 = target.BreakpointCreateByName ("stopper", 'a.out')
        self.assertTrue(break1, VALID_BREAKPOINT)
        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break1)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint 1.")

        queue_submittor_1 = lldb.SBQueue()
        queue_performer_1 = lldb.SBQueue()
        queue_performer_2 = lldb.SBQueue()
        queue_performer_3 = lldb.SBQueue()
        for idx in range (0, process.GetNumQueues()):
          q = process.GetQueueAtIndex(idx)
          if q.GetName() == "com.apple.work_submittor_1":
            queue_submittor_1 = q
          if q.GetName() == "com.apple.work_performer_1":
            queue_performer_1 = q
          if q.GetName() == "com.apple.work_performer_2":
            queue_performer_2 = q
          if q.GetName() == "com.apple.work_performer_3":
            queue_performer_3 = q

        self.assertTrue(queue_submittor_1.IsValid() and queue_performer_1.IsValid() and queue_performer_2.IsValid() and queue_performer_3.IsValid(), "Got all four expected queues: %s %s %s %s" % (queue_submittor_1.IsValid(), queue_performer_1.IsValid(), queue_performer_2.IsValid(), queue_performer_3.IsValid()))

        self.check_queue_for_valid_queue_id (queue_submittor_1)
        self.check_queue_for_valid_queue_id (queue_performer_1)
        self.check_queue_for_valid_queue_id (queue_performer_2)
        self.check_queue_for_valid_queue_id (queue_performer_3)

        self.check_number_of_threads_owned_by_queue (queue_submittor_1, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_1, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_2, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_3, 4)

        self.check_queue_kind (queue_submittor_1, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_1, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_2, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_3, lldb.eQueueKindConcurrent)
        
        self.check_queues_threads_match_queue (queue_submittor_1)
        self.check_queues_threads_match_queue (queue_performer_1)
        self.check_queues_threads_match_queue (queue_performer_2)
        self.check_queues_threads_match_queue (queue_performer_3)



        # We have threads running with all the different dispatch QoS service
        # levels - find those threads and check that we can get the correct
        # QoS name for each of them.

        user_initiated_thread = lldb.SBThread()
        user_interactive_thread = lldb.SBThread()
        utility_thread = lldb.SBThread()
        unspecified_thread = lldb.SBThread()
        background_thread = lldb.SBThread()
        for th in process.threads:
            if th.GetName() == "user initiated QoS":
                user_initiated_thread = th
            if th.GetName() == "user interactive QoS":
                user_interactive_thread = th
            if th.GetName() == "utility QoS":
                utility_thread = th
            if th.GetName() == "unspecified QoS":
                unspecified_thread = th
            if th.GetName() == "background QoS":
                background_thread = th

        self.assertTrue(user_initiated_thread.IsValid(), "Found user initiated QoS thread")
        self.assertTrue(user_interactive_thread.IsValid(), "Found user interactive QoS thread")
        self.assertTrue(utility_thread.IsValid(), "Found utility QoS thread")
        self.assertTrue(unspecified_thread.IsValid(), "Found unspecified QoS thread")
        self.assertTrue(background_thread.IsValid(), "Found background QoS thread")

        stream = lldb.SBStream()
        self.assertTrue(user_initiated_thread.GetInfoItemByPathAsString("requested_qos.printable_name", stream), "Get QoS printable string for user initiated QoS thread")
        self.assertTrue(stream.GetData() == "User Initiated", "user initiated QoS thread name is valid")
        stream.Clear()
        self.assertTrue(user_interactive_thread.GetInfoItemByPathAsString("requested_qos.printable_name", stream), "Get QoS printable string for user interactive QoS thread")
        self.assertTrue(stream.GetData() == "User Interactive", "user interactive QoS thread name is valid")
        stream.Clear()
        self.assertTrue(utility_thread.GetInfoItemByPathAsString("requested_qos.printable_name", stream), "Get QoS printable string for utility QoS thread")
        self.assertTrue(stream.GetData() == "Utility", "utility QoS thread name is valid")
        stream.Clear()
        self.assertTrue(unspecified_thread.GetInfoItemByPathAsString("requested_qos.printable_name", stream), "Get QoS printable string for unspecified QoS thread")
        self.assertTrue(stream.GetData() == "User Initiated", "unspecified QoS thread name is valid")
        stream.Clear()
        self.assertTrue(background_thread.GetInfoItemByPathAsString("requested_qos.printable_name", stream), "Get QoS printable string for background QoS thread")
        self.assertTrue(stream.GetData() == "Background", "background QoS thread name is valid")

    def queues_with_libBacktraceRecording(self):
        """Test queues inspection SB APIs with libBacktraceRecording present."""
        exe = os.path.join(os.getcwd(), "a.out")

        if not os.path.isfile('/Applications/Xcode.app/Contents/Developer/usr/lib/libBacktraceRecording.dylib'):
          self.skipTest ("Skipped because libBacktraceRecording.dylib was present on the system.")
          
        if not os.path.isfile('/usr/lib/system/introspection/libdispatch.dylib'):
          self.skipTest ("Skipped because introspection libdispatch dylib is not present.")
          
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec (self.main_source)

        break1 = target.BreakpointCreateByName ("stopper", 'a.out')
        self.assertTrue(break1, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, ['DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/usr/lib/libBacktraceRecording.dylib', 'DYLD_LIBRARY_PATH=/usr/lib/system/introspection'], self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break1)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint 1.")

        libbtr_module_filespec = lldb.SBFileSpec("libBacktraceRecording.dylib")
        libbtr_module = target.FindModule (libbtr_module_filespec)
        if not libbtr_module.IsValid():
          self.skipTest ("Skipped because libBacktraceRecording.dylib was not loaded into the process.")

        self.assertTrue(process.GetNumQueues() >= 4, "Found the correct number of queues.")

        queue_submittor_1 = lldb.SBQueue()
        queue_performer_1 = lldb.SBQueue()
        queue_performer_2 = lldb.SBQueue()
        queue_performer_3 = lldb.SBQueue()
        for idx in range (0, process.GetNumQueues()):
          q = process.GetQueueAtIndex(idx)
          if q.GetName() == "com.apple.work_submittor_1":
            queue_submittor_1 = q
          if q.GetName() == "com.apple.work_performer_1":
            queue_performer_1 = q
          if q.GetName() == "com.apple.work_performer_2":
            queue_performer_2 = q
          if q.GetName() == "com.apple.work_performer_3":
            queue_performer_3 = q

        self.assertTrue(queue_submittor_1.IsValid() and queue_performer_1.IsValid() and queue_performer_2.IsValid() and queue_performer_3.IsValid(), "Got all four expected queues: %s %s %s %s" % (queue_submittor_1.IsValid(), queue_performer_1.IsValid(), queue_performer_2.IsValid(), queue_performer_3.IsValid()))

        self.check_queue_for_valid_queue_id (queue_submittor_1)
        self.check_queue_for_valid_queue_id (queue_performer_1)
        self.check_queue_for_valid_queue_id (queue_performer_2)
        self.check_queue_for_valid_queue_id (queue_performer_3)

        self.check_running_and_pending_items_on_queue (queue_submittor_1, 1, 0)
        self.check_running_and_pending_items_on_queue (queue_performer_1, 1, 3)
        self.check_running_and_pending_items_on_queue (queue_performer_2, 1, 9999)
        self.check_running_and_pending_items_on_queue (queue_performer_3, 4, 0)
       
        self.check_number_of_threads_owned_by_queue (queue_submittor_1, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_1, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_2, 1)
        self.check_number_of_threads_owned_by_queue (queue_performer_3, 4)

        self.check_queue_kind (queue_submittor_1, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_1, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_2, lldb.eQueueKindSerial)
        self.check_queue_kind (queue_performer_3, lldb.eQueueKindConcurrent)
        

        self.check_queues_threads_match_queue (queue_submittor_1)
        self.check_queues_threads_match_queue (queue_performer_1)
        self.check_queues_threads_match_queue (queue_performer_2)
        self.check_queues_threads_match_queue (queue_performer_3)

        self.assertTrue(queue_performer_2.GetPendingItemAtIndex(0).IsValid(), "queue 2's pending item #0 is valid")
        self.assertTrue(queue_performer_2.GetPendingItemAtIndex(0).GetAddress().GetSymbol().GetName() == "doing_the_work_2", "queue 2's pending item #0 should be doing_the_work_2")
        self.assertTrue(queue_performer_2.GetNumPendingItems() == 9999, "verify that queue 2 still has 9999 pending items")
        self.assertTrue(queue_performer_2.GetPendingItemAtIndex(9998).IsValid(), "queue 2's pending item #9998 is valid")
        self.assertTrue(queue_performer_2.GetPendingItemAtIndex(9998).GetAddress().GetSymbol().GetName() == "doing_the_work_2", "queue 2's pending item #0 should be doing_the_work_2")
        self.assertTrue(queue_performer_2.GetPendingItemAtIndex(9999).IsValid() == False, "queue 2's pending item #9999 is invalid")
