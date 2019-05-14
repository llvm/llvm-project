"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb


def fuzz_obj(obj):
    obj.IsValid()
    obj.GetName()
    obj.FindSubSection("hello_section_name")
    obj.GetNumSubSections()
    obj.GetSubSectionAtIndex(600)
    obj.GetFileAddress()
    obj.GetByteSize()
    obj.GetFileOffset()
    obj.GetFileByteSize()
    obj.GetSectionData(1000, 100)
    obj.GetSectionType()
    obj.GetDescription(lldb.SBStream())
    for subsec in obj:
        s = str(subsec)
    len(obj)
