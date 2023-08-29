%extend lldb::SBThreadCollection {
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

    def __iter__(self):
        '''Iterate over all threads in a lldb.SBThreadCollection object.'''
        return lldb_iter(self, 'GetSize', 'GetThreadAtIndex')

    def __len__(self):
        '''Return the number of threads in a lldb.SBThreadCollection object.'''
        return self.GetSize()
    %}
#endif
}
