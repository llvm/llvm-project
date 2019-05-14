//===-- SWIG Interface for SBBlock ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a lexical block. SBFunction contains SBBlock(s)."
) SBBlock;
class SBBlock
{
public:

    SBBlock ();

    SBBlock (const lldb::SBBlock &rhs);

    ~SBBlock ();

    %feature("docstring",
    "Does this block represent an inlined function?"
    ) IsInlined;
    bool
    IsInlined () const;

    bool
    IsValid () const;

    explicit operator bool() const;

    %feature("docstring", "
    Get the function name if this block represents an inlined function;
    otherwise, return None.") GetInlinedName;
    const char *
    GetInlinedName () const;

    %feature("docstring", "
    Get the call site file if this block represents an inlined function;
    otherwise, return an invalid file spec.") GetInlinedCallSiteFile;
    lldb::SBFileSpec
    GetInlinedCallSiteFile () const;

    %feature("docstring", "
    Get the call site line if this block represents an inlined function;
    otherwise, return 0.") GetInlinedCallSiteLine;
    uint32_t
    GetInlinedCallSiteLine () const;

    %feature("docstring", "
    Get the call site column if this block represents an inlined function;
    otherwise, return 0.") GetInlinedCallSiteColumn;
    uint32_t
    GetInlinedCallSiteColumn () const;

    %feature("docstring", "Get the parent block.") GetParent;
    lldb::SBBlock
    GetParent ();

    %feature("docstring", "Get the inlined block that is or contains this block.") GetContainingInlinedBlock;
    lldb::SBBlock
    GetContainingInlinedBlock ();

    %feature("docstring", "Get the sibling block for this block.") GetSibling;
    lldb::SBBlock
    GetSibling ();

    %feature("docstring", "Get the first child block.") GetFirstChild;
    lldb::SBBlock
    GetFirstChild ();

    uint32_t
    GetNumRanges ();

    lldb::SBAddress
    GetRangeStartAddress (uint32_t idx);

    lldb::SBAddress
    GetRangeEndAddress (uint32_t idx);

    uint32_t
    GetRangeIndexForBlockAddress (lldb::SBAddress block_addr);

    bool
    GetDescription (lldb::SBStream &description);

    lldb::SBValueList
    GetVariables (lldb::SBFrame& frame,
                  bool arguments,
                  bool locals,
                  bool statics,
                  lldb::DynamicValueType use_dynamic);

     lldb::SBValueList
     GetVariables (lldb::SBTarget& target,
                   bool arguments,
                   bool locals,
                   bool statics);

    %pythoncode %{
        def get_range_at_index(self, idx):
            if idx < self.GetNumRanges():
                return [self.GetRangeStartAddress(idx), self.GetRangeEndAddress(idx)]
            return []

        class ranges_access(object):
            '''A helper object that will lazily hand out an array of lldb.SBAddress that represent address ranges for a block.'''
            def __init__(self, sbblock):
                self.sbblock = sbblock

            def __len__(self):
                if self.sbblock:
                    return int(self.sbblock.GetNumRanges())
                return 0

            def __getitem__(self, key):
                count = len(self)
                if type(key) is int:
                    return self.sbblock.get_range_at_index (key);
                if isinstance(key, SBAddress):
                    range_idx = self.sbblock.GetRangeIndexForBlockAddress(key);
                    if range_idx < len(self):
                        return [self.sbblock.GetRangeStartAddress(range_idx), self.sbblock.GetRangeEndAddress(range_idx)]
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None

        def get_ranges_access_object(self):
            '''An accessor function that returns a ranges_access() object which allows lazy block address ranges access.'''
            return self.ranges_access (self)

        def get_ranges_array(self):
            '''An accessor function that returns an array object that contains all ranges in this block object.'''
            if not hasattr(self, 'ranges_array'):
                self.ranges_array = []
                for idx in range(self.num_ranges):
                    self.ranges_array.append ([self.GetRangeStartAddress(idx), self.GetRangeEndAddress(idx)])
            return self.ranges_array

        def get_call_site(self):
            return declaration(self.GetInlinedCallSiteFile(), self.GetInlinedCallSiteLine(), self.GetInlinedCallSiteColumn())

        __swig_getmethods__["parent"] = GetParent
        if _newclass: parent = property(GetParent, None, doc='''A read only property that returns the same result as GetParent().''')

        __swig_getmethods__["first_child"] = GetFirstChild
        if _newclass: first_child = property(GetFirstChild, None, doc='''A read only property that returns the same result as GetFirstChild().''')

        __swig_getmethods__["call_site"] = get_call_site
        if _newclass: call_site = property(get_call_site, None, doc='''A read only property that returns a lldb.declaration object that contains the inlined call site file, line and column.''')

        __swig_getmethods__["sibling"] = GetSibling
        if _newclass: sibling = property(GetSibling, None, doc='''A read only property that returns the same result as GetSibling().''')

        __swig_getmethods__["name"] = GetInlinedName
        if _newclass: name = property(GetInlinedName, None, doc='''A read only property that returns the same result as GetInlinedName().''')

        __swig_getmethods__["inlined_block"] = GetContainingInlinedBlock
        if _newclass: inlined_block = property(GetContainingInlinedBlock, None, doc='''A read only property that returns the same result as GetContainingInlinedBlock().''')

        __swig_getmethods__["range"] = get_ranges_access_object
        if _newclass: range = property(get_ranges_access_object, None, doc='''A read only property that allows item access to the address ranges for a block by integer (range = block.range[0]) and by lldb.SBAdddress (find the range that contains the specified lldb.SBAddress like "pc_range = lldb.frame.block.range[frame.addr]").''')

        __swig_getmethods__["ranges"] = get_ranges_array
        if _newclass: ranges = property(get_ranges_array, None, doc='''A read only property that returns a list() object that contains all of the address ranges for the block.''')

        __swig_getmethods__["num_ranges"] = GetNumRanges
        if _newclass: num_ranges = property(GetNumRanges, None, doc='''A read only property that returns the same result as GetNumRanges().''')
    %}

};

} // namespace lldb
