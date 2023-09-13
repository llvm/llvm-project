%extend lldb::SBBreakpointList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
        '''Return the number of breakpoints in a lldb.SBBreakpointList object.'''
        return self.GetSize()

    def __iter__(self):
        '''Iterate over all breakpoints in a lldb.SBBreakpointList object.'''
        return lldb_iter(self, 'GetSize', 'GetBreakpointAtIndex')
    %}
#endif
}
