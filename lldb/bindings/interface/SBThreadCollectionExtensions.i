%extend lldb::SBThreadCollection {
#ifdef SWIGPYTHON
    %pythoncode%{

    def __iter__(self):
        '''Iterate over all threads in a lldb.SBThreadCollection object.'''
        return lldb_iter(self, 'GetSize', 'GetThreadAtIndex')

    def __len__(self):
        '''Return the number of threads in a lldb.SBThreadCollection object.'''
        return self.GetSize()
    %}
#endif
}
