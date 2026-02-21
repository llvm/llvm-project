STRING_EXTENSION_OUTSIDE(SBFileSpecList)

%extend lldb::SBFileSpecList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of FileSpec in a lldb.SBFileSpecList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all FileSpecs in a lldb.SBFileSpecList object.'''
      return lldb_iter(self, 'GetSize', 'GetFileSpecAtIndex')

    def __getitem__(self, idx):
      '''Get the FileSpec at a given index in an lldb.SBFileSpecList object.'''
      if not isinstance(idx, int):
        raise TypeError("unsupported index type: %s" % type(idx))
      count = len(self)
      if not (-count <= idx < count):
        raise IndexError("list index out of range")
      idx %= count
      return self.GetFileSpecAtIndex(idx)
    %}
#endif
}
