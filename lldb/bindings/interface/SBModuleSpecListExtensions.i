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

    def __getitem__(self, index):
      '''Get an lldb.SBModuleSpec at a given index, an invalid SBModuleSpec will be returned if the index is invalid.'''
      return self.GetSpecAtIndex(index) 
    %}
#endif
}
