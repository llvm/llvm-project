STRING_EXTENSION_OUTSIDE(SBCompileUnit)

%extend lldb::SBCompileUnit {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all line entries in a lldb.SBCompileUnit object.'''
            return lldb_iter(self, 'GetNumLineEntries', 'GetLineEntryAtIndex')

        def __len__(self):
            '''Return the number of line entries in a lldb.SBCompileUnit
            object.'''
            return self.GetNumLineEntries()

        file = property(GetFileSpec, None, doc='''A read only property that returns the same result an lldb object that represents the source file (lldb.SBFileSpec) for the compile unit.''')
        num_line_entries = property(GetNumLineEntries, None, doc='''A read only property that returns the number of line entries in a compile unit as an integer.''')
    %}
#endif
}
