%extend lldb::SBProcessInfoList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of process info in a lldb.SBProcessInfoListExtensions object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the process info in a lldb.SBProcessInfoListExtensions object.'''
      return lldb_iter(self, 'GetSize', 'GetProcessInfoAtIndex')

    def __getitem__(self, idx):
      '''Get the process info at a given index in an lldb.SBProcessInfoList object.'''
      if not isinstance(idx, int):
        raise TypeError("unsupported index type: %s" % type(idx))
      count = len(self)
      if not (-count <= idx < count):
        raise IndexError("list index out of range")
      idx %= count
      return self.GetProcessInfoAtIndex(idx)
    %}
#endif
}
