"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetName()
    obj.GetMangledName()
    obj.GetInstructions(lldb.SBTarget())
    obj.GetStartAddress()
    obj.GetEndAddress()
    obj.GetPrologueByteSize()
    obj.GetType()
    obj.GetDescription(lldb.SBStream())
