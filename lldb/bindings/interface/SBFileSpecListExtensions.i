STRING_EXTENSION_OUTSIDE(SBFileSpecList)

%extend lldb::SBFileSpecList {
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
      '''Return the number of FileSpec in a lldb.SBFileSpecList object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all FileSpecs in a lldb.SBFileSpecList object.'''
      return lldb_iter(self, 'GetSize', 'GetFileSpecAtIndex')
    %}
#endif
}
