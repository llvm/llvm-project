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
    %}
#endif
}
