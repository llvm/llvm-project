"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.IsValid()
    obj.GetName()
    obj.SetEnabled(True)
    obj.IsEnabled()
    obj.SetOneShot(True)
    obj.IsOneShot()
    obj.SetIgnoreCount(1)
    obj.GetIgnoreCount()
    obj.SetCondition("1 == 2")
    obj.GetCondition()
    obj.SetAutoContinue(False)
    obj.GetAutoContinue()
    obj.SetThreadID(0x1234)
    obj.GetThreadID()
    obj.SetThreadIndex(10)
    obj.GetThreadIndex()
    obj.SetThreadName("AThread")
    obj.GetThreadName()
    obj.SetQueueName("AQueue")
    obj.GetQueueName()
    obj.SetScriptCallbackFunction("AFunction")
    commands = lldb.SBStringList()
    obj.SetCommandLineCommands(commands)
    obj.GetCommandLineCommands(commands)
    obj.SetScriptCallbackBody("Insert Python Code here")
    obj.GetAllowList()
    obj.SetAllowList(False)
    obj.GetAllowDelete()
    obj.SetAllowDelete(False)
    obj.GetAllowDisable()
    obj.SetAllowDisable(False)
    stream = lldb.SBStream()
    obj.GetDescription(stream)
