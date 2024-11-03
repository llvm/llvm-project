STRING_EXTENSION_OUTSIDE(SBFileSpec)

%extend lldb::SBFileSpec {
#ifdef SWIGPYTHON
    %pythoncode %{
        fullpath = property(str, None, doc='''A read only property that returns the fullpath as a python string.''')
        basename = property(GetFilename, None, doc='''A read only property that returns the path basename as a python string.''')
        dirname = property(GetDirectory, None, doc='''A read only property that returns the path directory name as a python string.''')
        exists = property(Exists, None, doc='''A read only property that returns a boolean value that indicates if the file exists.''')
    %}
#endif
}
