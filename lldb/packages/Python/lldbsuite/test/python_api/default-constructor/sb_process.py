"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetTarget()
    obj.GetByteOrder()
    obj.PutSTDIN("my data")
    obj.GetSTDOUT(6)
    obj.GetSTDERR(6)
    event = lldb.SBEvent()
    obj.ReportEventState(event, None)
    obj.AppendEventStateReport(event, lldb.SBCommandReturnObject())
    error = lldb.SBError()
    obj.RemoteAttachToProcessWithID(123, error)
    obj.RemoteLaunch(None, None, None, None, None, None, 0, False, error)
    obj.GetNumThreads()
    obj.GetThreadAtIndex(0)
    obj.GetThreadByID(0)
    obj.GetSelectedThread()
    obj.SetSelectedThread(lldb.SBThread())
    obj.SetSelectedThreadByID(0)
    obj.GetState()
    obj.GetExitStatus()
    obj.GetExitDescription()
    obj.GetProcessID()
    obj.GetAddressByteSize()
    obj.Destroy()
    obj.Continue()
    obj.Stop()
    obj.Kill()
    obj.Detach()
    obj.Signal(7)
    obj.ReadMemory(0x0000ffff, 10, error)
    obj.WriteMemory(0x0000ffff, "hi data", error)
    obj.ReadCStringFromMemory(0x0, 128, error)
    obj.ReadUnsignedFromMemory(0xff, 4, error)
    obj.ReadPointerFromMemory(0xff, error)
    obj.GetBroadcaster()
    obj.GetDescription(lldb.SBStream())
    obj.LoadImage(lldb.SBFileSpec(), error)
    obj.UnloadImage(0)
    obj.Clear()
    obj.GetNumSupportedHardwareWatchpoints(error)
    for thread in obj:
        s = str(thread)
