//===-- SWIG Interface for SBMemoryRegionInfo -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"API clients can get information about memory regions in processes."
) SBMemoryRegionInfo;

class SBMemoryRegionInfo
{
public:

    SBMemoryRegionInfo ();

    SBMemoryRegionInfo (const lldb::SBMemoryRegionInfo &rhs);

    SBMemoryRegionInfo::SBMemoryRegionInfo(const char *name, lldb::addr_t begin,
    lldb::addr_t end, uint32_t permissions, bool mapped, bool stack_memory);

    ~SBMemoryRegionInfo ();

    void
    Clear();

    lldb::addr_t
    GetRegionBase ();

    lldb::addr_t
    GetRegionEnd ();

    bool
    IsReadable ();

    bool
    IsWritable ();

    bool
    IsExecutable ();

    bool
    IsMapped ();

    const char *
    GetName ();

    %feature("autodoc", "
        GetRegionEnd(SBMemoryRegionInfo self) -> lldb::addr_t
        Returns whether this memory region has a list of modified (dirty)
        pages available or not.  When calling GetNumDirtyPages(), you will
        have 0 returned for both \"dirty page list is not known\" and 
        \"empty dirty page list\" (that is, no modified pages in this
        memory region).  You must use this method to disambiguate.") HasDirtyMemoryPageList;
    bool 
    HasDirtyMemoryPageList();

    %feature("autodoc", "
        GetNumDirtyPages(SBMemoryRegionInfo self) -> uint32_t
        Return the number of dirty (modified) memory pages in this
        memory region, if available.  You must use the 
        SBMemoryRegionInfo::HasDirtyMemoryPageList() method to
        determine if a dirty memory list is available; it will depend
        on the target system can provide this information.") GetNumDirtyPages;
    uint32_t 
    GetNumDirtyPages();

    %feature("autodoc", "
        GetDirtyPageAddressAtIndex(SBMemoryRegionInfo self, uint32_t idx) -> lldb::addr_t
        Return the address of a modified, or dirty, page of memory.
        If the provided index is out of range, or this memory region 
        does not have dirty page information, LLDB_INVALID_ADDRESS 
        is returned.") GetDirtyPageAddressAtIndex;
    addr_t 
    GetDirtyPageAddressAtIndex(uint32_t idx);

    %feature("autodoc", "
        GetPageSize(SBMemoryRegionInfo self) -> int
        Return the size of pages in this memory region.  0 will be returned
        if this information was unavailable.") GetPageSize();
    int
    GetPageSize();

    bool
    operator == (const lldb::SBMemoryRegionInfo &rhs) const;

    bool
    operator != (const lldb::SBMemoryRegionInfo &rhs) const;

    bool
    GetDescription (lldb::SBStream &description);

    STRING_EXTENSION(SBMemoryRegionInfo)
};

} // namespace lldb
