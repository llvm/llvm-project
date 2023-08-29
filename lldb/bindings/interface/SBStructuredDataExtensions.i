STRING_EXTENSION_OUTSIDE(SBStructuredData)

%extend lldb::SBStructuredData {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
    def __eq__(self, other):
      return not self.__ne__(other)

    def __int__(self):
      return self.GetSignedInteger()

    def __hex__(self):
      return hex(self.GetSignedInteger())

    def __oct__(self):
      return oct(self.GetSignedInteger())

    def __len__(self):
      '''Return the number of element in a lldb.SBStructuredData object.'''
      return self.GetSize()

    def __iter__(self):
        '''Iterate over all the elements in a lldb.SBStructuredData object.'''
        return lldb_iter(self, 'GetSize', 'GetItemAtIndex')
    %}
#endif
}
