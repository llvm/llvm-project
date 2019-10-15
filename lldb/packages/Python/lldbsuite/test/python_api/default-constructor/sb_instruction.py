"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetAddress()
    obj.GetByteSize()
    obj.DoesBranch()
    try:
        obj.Print(None)
    except Exception:
        pass
    obj.GetDescription(lldb.SBStream())
    obj.EmulateWithFrame(lldb.SBFrame(), 0)
    obj.DumpEmulation("armv7")
    obj.TestEmulation(lldb.SBStream(), "my-file")
