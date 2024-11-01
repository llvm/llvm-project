%extend lldb::SBStringList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __iter__(self):
        '''Iterate over all strings in a lldb.SBStringList object.'''
        return lldb_iter(self, 'GetSize', 'GetStringAtIndex')

    def __len__(self):
        '''Return the number of strings in a lldb.SBStringList object.'''
        return self.GetSize()
    %}
#endif
}
