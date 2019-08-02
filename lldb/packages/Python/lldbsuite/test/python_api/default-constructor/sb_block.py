"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.IsInlined()
    obj.GetInlinedName()
    obj.GetInlinedCallSiteFile()
    obj.GetInlinedCallSiteLine()
    obj.GetInlinedCallSiteColumn()
    obj.GetParent()
    obj.GetSibling()
    obj.GetFirstChild()
    obj.GetDescription(lldb.SBStream())
