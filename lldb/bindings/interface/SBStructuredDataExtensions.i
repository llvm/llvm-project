STRING_EXTENSION_OUTSIDE(SBStructuredData)

%extend lldb::SBStructuredData {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __int__(self):
      return self.GetSignedInteger()

    def __len__(self):
      '''Return the number of element in a lldb.SBStructuredData object.'''
      return self.GetSize()

    def __iter__(self):
        '''Iterate over all the elements in a lldb.SBStructuredData object.'''
        return lldb_iter(self, 'GetSize', 'GetItemAtIndex')
    %}
#endif
}
