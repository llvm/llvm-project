"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.GetStartAddress()
    obj.GetEndAddress()
    obj.GetFileSpec()
    obj.GetLine()
    obj.GetColumn()
    obj.GetDescription(lldb.SBStream())
