%extend lldb::SBAddressRangeList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of address ranges in a lldb.SBAddressRangeList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the address ranges in a lldb.SBAddressRangeList object.'''
      return lldb_iter(self, 'GetSize', 'GetAddressRangeAtIndex')

    def __getitem__(self, idx):
      '''Get the address range at a given index in an lldb.SBAddressRangeList object.'''
      if not isinstance(idx, int):
        raise TypeError("unsupported index type: %s" % type(idx))
      count = len(self)
      if not (-count <= idx < count):
        raise IndexError("list index out of range")
      idx %= count
      return self.GetAddressRangeAtIndex(idx)

    def __repr__(self):
      import lldb
      stream = lldb.SBStream()
      self.GetDescription(stream, lldb.target if lldb.target else lldb.SBTarget())
      return stream.GetData()
    %}
#endif
}
