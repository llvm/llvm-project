"""
Test lldb Python event APIs.
"""

from __future__ import print_function



import os, time
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

@skipIfDarwin  # llvm.org/pr25924, sometimes generating SIGSEGV
@skipIfLinux   # llvm.org/pr25924, sometimes generating SIGSEGV
class EventAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Find the line number of function "c" here.')

    @add_test_categories(['pyapi'])
    @expectedFailureLinux("llvm.org/pr23730") # Flaky, fails ~1/10 cases
    def test_listen_for_and_print_event(self):
        """Exercise SBEvent API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        listener = lldb.SBListener("my listener")

        # Now launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        process = target.Launch (listener, 
                                 None,      # argv
                                 None,      # envp
                                 None,      # stdin_path
                                 None,      # stdout_path
                                 None,      # stderr_path
                                 None,      # working directory
                                 0,         # launch flags
                                 False,     # Stop at entry
                                 error)     # error

        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)

        # Create an empty event object.
        event = lldb.SBEvent()

        traceOn = self.TraceOn()
        if traceOn:
            lldbutil.print_stacktraces(process)

        # Create MyListeningThread class to wait for any kind of event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 4 times to retrieve any kind of event.
                # After that, the thread exits.
                while not count > 3:
                    if traceOn:
                        print("Try wait for event...")
                    if listener.WaitForEvent(5, event):
                        if traceOn:
                            desc = lldbutil.get_description(event)
                            print("Event description:", desc)
                            print("Event data flavor:", event.GetDataFlavor())
                            print("Process state:", lldbutil.state_type_to_str(process.GetState()))
                            print()
                    else:
                        if traceOn:
                            print("timeout occurred waiting for event...")
                    count = count + 1
                return

        # Let's start the listening thread to retrieve the events.
        my_thread = MyListeningThread()
        my_thread.start()

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        process.Continue()

        # Use Python API to kill the process.  The listening thread should be
        # able to receive the state changed event, too.
        process.Kill()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

    @add_test_categories(['pyapi'])
    @expectedFlakeyLinux("llvm.org/pr23730") # Flaky, fails ~1/100 cases
    def test_wait_for_event(self):
        """Exercise SBListener.WaitForEvent() API."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the debugger listener.
        listener = self.dbg.GetListener()

        # Now launch the process, and do not stop at entry point.
        error = lldb.SBError()
        process = target.Launch (listener, 
                                 None,      # argv
                                 None,      # envp
                                 None,      # stdin_path
                                 None,      # stdout_path
                                 None,      # stderr_path
                                 None,      # working directory
                                 0,         # launch flags
                                 False,     # Stop at entry
                                 error)     # error
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event, "Event should not be valid initially")

        # Create MyListeningThread to wait for any kind of event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 3 times to retrieve any kind of event.
                while not count > 3:
                    if listener.WaitForEvent(5, event):
                        #print("Got a valid event:", event)
                        #print("Event data flavor:", event.GetDataFlavor())
                        #print("Event type:", lldbutil.state_type_to_str(event.GetType()))
                        return
                    count = count + 1
                    print("Timeout: listener.WaitForEvent")

                return

        # Use Python API to kill the process.  The listening thread should be
        # able to receive a state changed event.
        process.Kill()

        # Let's start the listening thread to retrieve the event.
        my_thread = MyListeningThread()
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        self.assertTrue(event,
                        "My listening thread successfully received an event")

    @skipIfFreeBSD # llvm.org/pr21325
    @add_test_categories(['pyapi'])
    @expectedFailureLinux("llvm.org/pr23617")  # Flaky, fails ~1/10 cases
    @expectedFailureWindows("llvm.org/pr24778")
    def test_add_listener_to_broadcaster(self):
        """Exercise some SBBroadcaster APIs."""
        self.build()
        exe = os.path.join(os.getcwd(), "a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print("breakpoint:", breakpoint)
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        listener = lldb.SBListener("my listener")

        # Now launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        process = target.Launch (listener, 
                                 None,      # argv
                                 None,      # envp
                                 None,      # stdin_path
                                 None,      # stdout_path
                                 None,      # stderr_path
                                 None,      # working directory
                                 0,         # launch flags
                                 False,     # Stop at entry
                                 error)     # error

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event, "Event should not be valid initially")


        # The finite state machine for our custom listening thread, with an
        # initial state of None, which means no event has been received.
        # It changes to 'connected' after 'connected' event is received (for remote platforms)
        # It changes to 'running' after 'running' event is received (should happen only if the
        # currentstate is either 'None' or 'connected')
        # It changes to 'stopped' if a 'stopped' event is received (should happen only if the
        # current state is 'running'.)
        self.state = None

        # Create MyListeningThread to wait for state changed events.
        # By design, a "running" event is expected following by a "stopped" event.
        import threading
        class MyListeningThread(threading.Thread):
            def run(self):
                #print("Running MyListeningThread:", self)

                # Regular expression pattern for the event description.
                pattern = re.compile("data = {.*, state = (.*)}$")

                # Let's only try at most 6 times to retrieve our events.
                count = 0
                while True:
                    if listener.WaitForEvent(5, event):
                        desc = lldbutil.get_description(event)
                        #print("Event description:", desc)
                        match = pattern.search(desc)
                        if not match:
                            break;
                        if match.group(1) == 'connected':
                            # When debugging remote targets with lldb-server, we
                            # first get the 'connected' event.
                            self.context.assertTrue(self.context.state == None)
                            self.context.state = 'connected'
                            continue
                        elif match.group(1) == 'running':
                            self.context.assertTrue(self.context.state == None or self.context.state == 'connected')
                            self.context.state = 'running'
                            continue
                        elif match.group(1) == 'stopped':
                            self.context.assertTrue(self.context.state == 'running')
                            # Whoopee, both events have been received!
                            self.context.state = 'stopped'
                            break
                        else:
                            break
                    print("Timeout: listener.WaitForEvent")
                    count = count + 1
                    if count > 6:
                        break

                return

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        process.Continue()

        # Start the listening thread to receive the "running" followed by the
        # "stopped" events.
        my_thread = MyListeningThread()
        # Supply the enclosing context so that our listening thread can access
        # the 'state' variable.
        my_thread.context = self
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        # The final judgement. :-)
        self.assertTrue(self.state == 'stopped',
                        "Both expected state changed events received")
