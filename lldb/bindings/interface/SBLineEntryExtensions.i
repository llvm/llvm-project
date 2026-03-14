STRING_EXTENSION_OUTSIDE(SBLineEntry)

%extend lldb::SBLineEntry {
#ifdef SWIGPYTHON
    %pythoncode %{
        # operator== is a free function, which swig does not handle, so we inject
        # our own equality operator here
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            return self.GetLine()

        def __hex__(self):
            return self.GetStartAddress()
        file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this line entry.''')
        line = property(GetLine, None, doc='''A read only property that returns the 1 based line number for this line entry, a return value of zero indicates that no line information is available.''')
        column = property(GetColumn, None, doc='''A read only property that returns the 1 based column number for this line entry, a return value of zero indicates that no column information is available.''')
        addr = property(GetStartAddress, None, doc='''A read only property that returns an lldb object that represents the start address (lldb.SBAddress) for this line entry.''')
        end_addr = property(GetEndAddress, None, doc='''A read only property that returns an lldb object that represents the end address (lldb.SBAddress) for this line entry.''')
    %}
#endif
}
