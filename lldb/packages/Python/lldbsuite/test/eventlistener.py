import threading

import lldb
from lldbsuite.test.lldbtest import *

class EventListenerTestBase(TestBase):

    """
    Base class for lldb event listener tests.

    This class will setup and start an event listener for the test to use.

    If the event received matches the source broadcaster, the event is
    queued up in a list that the user can access later on.
    """
    NO_DEBUG_INFO_TESTCASE = True

    eBroadcastBitStopListenerThread = (1 << 0)
    events = []
    event_mask = None
    event_data_extractor = None

    def setUp(self):
        TestBase.setUp(self)

        self.src_broadcaster = self.dbg.GetBroadcaster()
        self.broadcaster = lldb.SBBroadcaster('lldb.test.broadcaster')
        self.listener = lldb.SBListener("lldb.test.listener")
        self.listener.StartListeningForEvents(self.broadcaster,
                                              self.eBroadcastBitStopListenerThread)

        self.src_broadcaster.AddListener(self.listener, self.event_mask)

        self.listener_thread = threading.Thread(target=self._fetch_events)
        self.listener_thread.start()

    def tearDown(self):
        # Broadcast a `eBroadcastBitStopListenerThread` event so the background
        # thread stops listening to events, then join the background thread.
        self.broadcaster.BroadcastEventByType(self.eBroadcastBitStopListenerThread)
        self.listener_thread.join()
        TestBase.tearDown(self)

    def _fetch_events(self):
        event = lldb.SBEvent()

        done = False
        while not done:
            if self.listener.GetNextEvent(event):
                event_mask = event.GetType();
                if event.BroadcasterMatchesRef(self.broadcaster):
                    if event_mask & self.eBroadcastBitStopListenerThread:
                        done = True;
                elif event.BroadcasterMatchesRef(self.src_broadcaster):
                    # NOTE: https://wiki.python.org/moin/FromFunctionToMethod
                    #
                    # When assigning the `event_data_extractor` function pointer
                    # to the `EventListenerTestBase` instance, it turns the
                    # function object into an instance method which subsequently
                    # passes `self` as an extra argument.

                    # However, because most of the event data extractor
                    # functions are static, passing the `self` argument makes
                    # the number of passed arguments exceeds the function definition

                    # This is why we unwrap the function from the instance
                    # method object calling `__func__` instead.
                    ret_args = self.event_data_extractor.__func__(event)
                    if not ret_args:
                        continue

                    self.events.append(ret_args)
