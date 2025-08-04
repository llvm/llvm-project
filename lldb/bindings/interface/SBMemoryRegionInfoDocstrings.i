%feature("docstring",
"API clients can get information about memory regions in processes.

For Python users, `len()` is overriden to output the size of the memory region in bytes.
For Python users, `str()` is overriden with the results of the GetDescription function-
        produces a formatted string that describes a memory range in the form: 
        [Hex start - Hex End) with associated permissions (RWX)"
) lldb::SBMemoryRegionInfo;

%feature("docstring", "
        Returns whether this memory region has a list of modified (dirty)
        pages available or not.  When calling GetNumDirtyPages(), you will
        have 0 returned for both \"dirty page list is not known\" and 
        \"empty dirty page list\" (that is, no modified pages in this
        memory region).  You must use this method to disambiguate."
) lldb::SBMemoryRegionInfo::HasDirtyMemoryPageList;

%feature("docstring", "
        Return the number of dirty (modified) memory pages in this
        memory region, if available.  You must use the 
        SBMemoryRegionInfo::HasDirtyMemoryPageList() method to
        determine if a dirty memory list is available; it will depend
        on the target system can provide this information."
) lldb::SBMemoryRegionInfo::GetNumDirtyPages;

%feature("docstring", "
        Return the address of a modified, or dirty, page of memory.
        If the provided index is out of range, or this memory region 
        does not have dirty page information, LLDB_INVALID_ADDRESS 
        is returned."
) lldb::SBMemoryRegionInfo::GetDirtyPageAddressAtIndex;

%feature("docstring", "
        Return the size of pages in this memory region.  0 will be returned
        if this information was unavailable."
) lldb::SBMemoryRegionInfo::GetPageSize();

%feature("docstring", "
        Takes an SBStream parameter to write output to,
        formatted [Hex start - Hex End) with associated permissions (RWX).
        If the function results false, no output will be written. 
        If results true, the output will be written to the stream.
        "
) lldb::SBMemoryRegionInfo::GetDescription;