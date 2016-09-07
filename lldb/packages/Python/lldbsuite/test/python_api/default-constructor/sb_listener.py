"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.AddEvent(lldb.SBEvent())
    obj.StartListeningForEvents(lldb.SBBroadcaster(), 0xffffffff)
    obj.StopListeningForEvents(lldb.SBBroadcaster(), 0xffffffff)
    event = lldb.SBEvent()
    broadcaster = lldb.SBBroadcaster()
    obj.WaitForEvent(5, event)
    obj.WaitForEventForBroadcaster(5, broadcaster, event)
    obj.WaitForEventForBroadcasterWithType(5, broadcaster, 0xffffffff, event)
    obj.PeekAtNextEvent(event)
    obj.PeekAtNextEventForBroadcaster(broadcaster, event)
    obj.PeekAtNextEventForBroadcasterWithType(broadcaster, 0xffffffff, event)
    obj.GetNextEvent(event)
    obj.GetNextEventForBroadcaster(broadcaster, event)
    obj.GetNextEventForBroadcasterWithType(broadcaster, 0xffffffff, event)
    obj.HandleBroadcastEvent(event)
