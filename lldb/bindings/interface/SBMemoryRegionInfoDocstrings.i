%feature("docstring",
"API clients can get information about memory regions in processes."
) lldb::SBMemoryRegionInfo;

%feature("autodoc", "
        GetRegionEnd(SBMemoryRegionInfo self) -> lldb::addr_t
        Returns whether this memory region has a list of modified (dirty)
        pages available or not.  When calling GetNumDirtyPages(), you will
        have 0 returned for both \"dirty page list is not known\" and 
        \"empty dirty page list\" (that is, no modified pages in this
        memory region).  You must use this method to disambiguate."
) lldb::SBMemoryRegionInfo::HasDirtyMemoryPageList;

%feature("autodoc", "
        GetNumDirtyPages(SBMemoryRegionInfo self) -> uint32_t
        Return the number of dirty (modified) memory pages in this
        memory region, if available.  You must use the 
        SBMemoryRegionInfo::HasDirtyMemoryPageList() method to
        determine if a dirty memory list is available; it will depend
        on the target system can provide this information."
) lldb::SBMemoryRegionInfo::GetNumDirtyPages;

%feature("autodoc", "
        GetDirtyPageAddressAtIndex(SBMemoryRegionInfo self, uint32_t idx) -> lldb::addr_t
        Return the address of a modified, or dirty, page of memory.
        If the provided index is out of range, or this memory region 
        does not have dirty page information, LLDB_INVALID_ADDRESS 
        is returned."
) lldb::SBMemoryRegionInfo::GetDirtyPageAddressAtIndex;

%feature("autodoc", "
        GetPageSize(SBMemoryRegionInfo self) -> int
        Return the size of pages in this memory region.  0 will be returned
        if this information was unavailable."
) lldb::SBMemoryRegionInfo::GetPageSize();
