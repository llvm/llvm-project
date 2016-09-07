"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetID()
    obj.ClearAllBreakpointSites()
    obj.FindLocationByAddress(sys.maxsize)
    obj.FindLocationIDByAddress(sys.maxsize)
    obj.FindLocationByID(0)
    obj.GetLocationAtIndex(0)
    obj.SetEnabled(True)
    obj.IsEnabled()
    obj.GetHitCount()
    obj.SetIgnoreCount(1)
    obj.GetIgnoreCount()
    obj.SetCondition("i >= 10")
    obj.GetCondition()
    obj.SetThreadID(0)
    obj.GetThreadID()
    obj.SetThreadIndex(0)
    obj.GetThreadIndex()
    obj.SetThreadName("worker thread")
    obj.GetThreadName()
    obj.SetQueueName("my queue")
    obj.GetQueueName()
    obj.SetScriptCallbackFunction(None)
    obj.SetScriptCallbackBody(None)
    obj.GetNumResolvedLocations()
    obj.GetNumLocations()
    obj.GetDescription(lldb.SBStream())
    for bp_loc in obj:
        s = str(bp_loc)
