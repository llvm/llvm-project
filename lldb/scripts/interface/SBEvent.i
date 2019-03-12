//===-- SWIG Interface for SBEvent ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBBroadcaster;

%feature("docstring",
"API clients can register to receive events.

For example, check out the following output:

Try wait for event...
Event description: 0x103d0bb70 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = running}
Event data flavor: Process::ProcessEventData
Process state: running

Try wait for event...
Event description: 0x103a700a0 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = stopped}
Event data flavor: Process::ProcessEventData
Process state: stopped

Try wait for event...
Event description: 0x103d0d4a0 Event: broadcaster = 0x1009c8410, type = 0x00000001, data = { process = 0x1009c8400 (pid = 21528), state = exited}
Event data flavor: Process::ProcessEventData
Process state: exited

Try wait for event...
timeout occurred waiting for event...

from test/python_api/event/TestEventspy:

    def do_listen_for_and_print_event(self):
        '''Create a listener and use SBEvent API to print the events received.'''
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Get a handle on the process's broadcaster.
        broadcaster = process.GetBroadcaster()

        # Create an empty event object.
        event = lldb.SBEvent()

        # Create a listener object and register with the broadcaster.
        listener = lldb.SBListener('my listener')
        rc = broadcaster.AddListener(listener, lldb.SBProcess.eBroadcastBitStateChanged)
        self.assertTrue(rc, 'AddListener successfully retruns')

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
                        print('Try wait for event...')
                    if listener.WaitForEventForBroadcasterWithType(5,
                                                                   broadcaster,
                                                                   lldb.SBProcess.eBroadcastBitStateChanged,
                                                                   event):
                        if traceOn:
                            desc = lldbutil.get_description(event))
                            print('Event description:', desc)
                            print('Event data flavor:', event.GetDataFlavor())
                            print('Process state:', lldbutil.state_type_to_str(process.GetState()))
                            print()
                    else:
                        if traceOn:
                            print 'timeout occurred waiting for event...'
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
") SBEvent;
class SBEvent
{
public:
    SBEvent();

    SBEvent (const lldb::SBEvent &rhs);
    
    %feature("autodoc",
    "__init__(self, int type, str data) -> SBEvent (make an event that contains a C string)"
    ) SBEvent;
    SBEvent (uint32_t event, const char *cstr, uint32_t cstr_len);

    ~SBEvent();

    bool
    IsValid() const;

    explicit operator bool() const;

    const char *
    GetDataFlavor ();

    uint32_t
    GetType () const;

    lldb::SBBroadcaster
    GetBroadcaster () const;

    const char *
    GetBroadcasterClass () const;

    bool
    BroadcasterMatchesRef (const lldb::SBBroadcaster &broadcaster);

    void
    Clear();

    static const char *
    GetCStringFromEvent (const lldb::SBEvent &event);

    bool
    GetDescription (lldb::SBStream &description) const;
};

} // namespace lldb
