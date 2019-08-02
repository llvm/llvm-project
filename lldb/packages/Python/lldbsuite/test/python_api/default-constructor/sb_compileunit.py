"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetFileSpec()
    obj.GetNumLineEntries()
    obj.GetLineEntryAtIndex(0xffffffff)
    obj.FindLineEntryIndex(0, 0xffffffff, None)
    obj.GetDescription(lldb.SBStream())
    len(obj)
    for line_entry in obj:
        s = str(line_entry)
