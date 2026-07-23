%extend lldb::SBThreadCollection {
#ifdef SWIGPYTHON
    %pythoncode%{

    def __iter__(self):
        '''Iterate over all threads in a lldb.SBThreadCollection object.'''
        return lldb_iter(self, 'GetSize', 'GetThreadAtIndex')

    def __len__(self):
        '''Return the number of threads in a lldb.SBThreadCollection object.'''
        return self.GetSize()

    def __getitem__(self, idx):
        '''Get the thread at a given index in an lldb.SBThreadCollection object.'''
        if not isinstance(idx, int):
            raise TypeError("unsupported index type: %s" % type(idx))
        count = len(self)
        if not (-count <= idx < count):
            raise IndexError("list index out of range")
        idx %= count
        return self.GetThreadAtIndex(idx)
    %}
#endif
}
