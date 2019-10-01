"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetID()
    obj.IsValid()
    obj.GetHardwareIndex()
    obj.GetWatchAddress()
    obj.GetWatchSize()
    obj.SetEnabled(True)
    obj.IsEnabled()
    obj.GetHitCount()
    obj.GetIgnoreCount()
    obj.SetIgnoreCount(5)
    obj.GetDescription(lldb.SBStream(), lldb.eDescriptionLevelVerbose)
    obj.SetCondition("shouldWeStop()")
    obj.GetCondition()
