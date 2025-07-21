% feature("docstring",
          "API clients can get information about memory regions in processes.")
        lldb::SBMemoryRegionInfo;

%feature("docstring", "
        Returns whether this memory region has a list of modified (dirty)
        pages available or not.  When calling GetNumDirtyPages(), you will
        have 0 returned for both \"dirty page list is not known\" and
        \"empty dirty page list\" (that is, no modified pages in this
        memory region).  You must use this method to disambiguate."
) lldb::SBMemoryRegionInfo::HasDirtyMemoryPageList;

% feature(
      "docstring",
      "
      Return the number of dirty(modified) memory pages in this memory region,
      if available.You must use the SBMemoryRegionInfo::HasDirtyMemoryPageList()
          method to determine if a dirty memory list is available;
      it will depend on the target system can provide this information."
      ) lldb::SBMemoryRegionInfo::GetNumDirtyPages;

% feature("docstring",
          "
          Return the address of a modified,
          or dirty, page of memory.If the provided index is out of range,
          or this memory region does not have dirty page information,
          LLDB_INVALID_ADDRESS is returned."
          ) lldb::SBMemoryRegionInfo::GetDirtyPageAddressAtIndex;

% feature("docstring", "
                       Return the size of pages in this memory region .0 will be
                           returned if this information was unavailable."
          ) lldb::SBMemoryRegionInfo::GetPageSize();

%feature("docstring", "
        takes a SBStream parameter 'description' where it will write the output to.
        it formats the memory region information into a string with Memory region info
        [Hex start - Hex End) and premission flags R/W/X
        returns a boolean value indicating success or failure

        alternative to using this method to find out the size of the memory region
        is to use the len() function on the SBMemoryRegionInfo object"
) lldb::SBMemoryRegionInfo::GetDescription;
