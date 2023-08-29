STRING_EXTENSION_OUTSIDE(SBDeclaration)

%extend lldb::SBDeclaration {
#ifdef SWIGPYTHON
        %pythoncode %{
            def __int__(self):
                pass

            def __hex__(self):
                pass

            def __oct__(self):
                pass

            def __len__(self):
                pass

            def __iter__(self):
                pass

            file = property(GetFileSpec, None, doc='''A read only property that returns an lldb object that represents the file (lldb.SBFileSpec) for this line entry.''')
            line = property(GetLine, None, doc='''A read only property that returns the 1 based line number for this line entry, a return value of zero indicates that no line information is available.''')
            column = property(GetColumn, None, doc='''A read only property that returns the 1 based column number for this line entry, a return value of zero indicates that no column information is available.''')
        %}
#endif
}
