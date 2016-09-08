"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetStopReason()
    obj.GetStopReasonDataCount()
    obj.GetStopReasonDataAtIndex(100)
    obj.GetStopDescription(256)
    obj.GetThreadID()
    obj.GetIndexID()
    obj.GetName()
    obj.GetQueueName()
    obj.StepOver(lldb.eOnlyDuringStepping)
    obj.StepInto(lldb.eOnlyDuringStepping)
    obj.StepOut()
    frame = lldb.SBFrame()
    obj.StepOutOfFrame(frame)
    obj.StepInstruction(True)
    filespec = lldb.SBFileSpec()
    obj.StepOverUntil(frame, filespec, 1234)
    obj.RunToAddress(0xabcd)
    obj.Suspend()
    obj.Resume()
    obj.IsSuspended()
    obj.GetNumFrames()
    obj.GetFrameAtIndex(200)
    obj.GetSelectedFrame()
    obj.SetSelectedFrame(999)
    obj.GetProcess()
    obj.GetDescription(lldb.SBStream())
    obj.Clear()
    for frame in obj:
        s = str(frame)
