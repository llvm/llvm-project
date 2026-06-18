%extend lldb::SBBreakpointList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
        '''Return the number of breakpoints in a lldb.SBBreakpointList object.'''
        return self.GetSize()

    def __iter__(self):
        '''Iterate over all breakpoints in a lldb.SBBreakpointList object.'''
        return lldb_iter(self, 'GetSize', 'GetBreakpointAtIndex')

    def __getitem__(self, idx):
        '''Get the breakpoint at a given index in an lldb.SBBreakpointList object.'''
        if not isinstance(idx, int):
            raise TypeError("unsupported index type: %s" % type(idx))
        count = len(self)
        if not (-count <= idx < count):
            raise IndexError("list index out of range")
        idx %= count
        return self.GetBreakpointAtIndex(idx)
    %}
#endif
}
