%extend lldb::SBStringList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __iter__(self):
        '''Iterate over all strings in a lldb.SBStringList object.'''
        return lldb_iter(self, 'GetSize', 'GetStringAtIndex')

    def __len__(self):
        '''Return the number of strings in a lldb.SBStringList object.'''
        return self.GetSize()

    def __getitem__(self, idx):
        '''Get the string at a given index in an lldb.SBStringList object.'''
        if not isinstance(idx, int):
            raise TypeError("unsupported index type: %s" % type(idx))
        count = len(self)
        if not (-count <= idx < count):
            raise IndexError("list index out of range")
        idx %= count
        return self.GetStringAtIndex(idx)
    %}
#endif
}
