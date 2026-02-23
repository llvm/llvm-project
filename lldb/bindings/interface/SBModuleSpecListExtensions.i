STRING_EXTENSION_OUTSIDE(SBModuleSpecList)

%extend lldb::SBModuleSpecList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of ModuleSpec in a lldb.SBModuleSpecList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all ModuleSpecs in a lldb.SBModuleSpecList object.'''
      return lldb_iter(self, 'GetSize', 'GetSpecAtIndex')

    def __getitem__(self, idx):
      '''Get the ModuleSpec at a given index in an lldb.SBModuleSpecList object.'''
      if not isinstance(idx, int):
        raise TypeError("unsupported index type: %s" % type(idx))
      count = len(self)
      if not (-count <= idx < count):
        raise IndexError("list index out of range")
      idx %= count
      return self.GetSpecAtIndex(idx)
    %}
#endif
}

