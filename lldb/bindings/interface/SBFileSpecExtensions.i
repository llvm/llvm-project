STRING_EXTENSION_OUTSIDE(SBFileSpec)

%extend lldb::SBFileSpec {
#ifdef SWIGPYTHON
    %pythoncode %{
        # operator== is a free function, which swig does not handle, so we inject
        # our own equality operator here
        def __eq__(self, other):
            return not self.__ne__(other)

        fullpath = property(str, None, doc='''A read only property that returns the fullpath as a python string.''')
        basename = property(GetFilename, None, doc='''A read only property that returns the path basename as a python string.''')
        dirname = property(GetDirectory, None, doc='''A read only property that returns the path directory name as a python string.''')
        exists = property(Exists, None, doc='''A read only property that returns a boolean value that indicates if the file exists.''')
    %}
#endif
}
