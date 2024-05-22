%extend lldb::SBMemoryRegionInfoList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of memory region info in a lldb.SBMemoryRegionInfoList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the memory regions in a lldb.SBMemoryRegionInfoList object.'''
      return lldb_iter(self, 'GetSize', 'GetMemoryRegionAtIndex')
    %}
#endif
}
