"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.BroadcastEventByType(lldb.eBreakpointEventTypeInvalidType, True)
    obj.BroadcastEvent(lldb.SBEvent(), False)
    listener = lldb.SBListener("fuzz_testing")
    obj.AddInitialEventsToListener(listener, 0xffffffff)
    obj.AddInitialEventsToListener(listener, 0)
    obj.AddListener(listener, 0xffffffff)
    obj.AddListener(listener, 0)
    obj.GetName()
    obj.EventTypeHasListeners(0)
    obj.RemoveListener(listener, 0xffffffff)
    obj.RemoveListener(listener, 0)
    obj.Clear()
