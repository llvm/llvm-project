%extend lldb::SBMemoryRegionInfoList {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
    def __eq__(self, other):
      return not self.__ne__(other)

    def __int__(self):
      pass

    def __hex__(self):
      pass

    def __oct__(self):
      pass

    def __len__(self):
      '''Return the number of memory region info in a lldb.SBMemoryRegionInfoList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the memory regions in a lldb.SBMemoryRegionInfoList object.'''
      return lldb_iter(self, 'GetSize', 'GetMemoryRegionAtIndex')
    %}
#endif
}
