"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.IsValid()
    obj.GetName()
    obj.GetExecutableFile()
    obj.GetProcessID()
    obj.GetUserID()
    obj.GetGroupID()
    obj.UserIDIsValid()
    obj.GroupIDIsValid()
    obj.GetEffectiveUserID()
    obj.GetEffectiveGroupID()
    obj.EffectiveUserIDIsValid()
    obj.EffectiveGroupIDIsValid()
    obj.GetParentProcessID()
