%extend lldb::SBMemoryRegionInfoList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of memory region info in a lldb.SBMemoryRegionInfoList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the memory regions in a lldb.SBMemoryRegionInfoList object.'''
      import lldb
      size = self.GetSize()
      for i in range(size):
        region = lldb.SBMemoryRegionInfo()
        self.GetMemoryRegionAtIndex(i, region)
        yield region

    def __getitem__(self, idx):
      '''Get the memory region at a given index in an lldb.SBMemoryRegionInfoList object.'''
      if not isinstance(idx, int):
        raise TypeError("unsupported index type: %s" % type(idx))
      count = len(self)
      if not (-count <= idx < count):
        raise IndexError("list index out of range")
      idx %= count
      import lldb
      region = lldb.SBMemoryRegionInfo()
      self.GetMemoryRegionAtIndex(idx, region)
      return region
    %}
#endif
}
