%extend lldb::SBScriptObject {
#ifdef SWIGPYTHON
    %pythoncode %{
        # operator== is a free function, which swig does not handle, so we inject
        # our own equality operator here
        def __eq__(self, other):
            return not self.__ne__(other)

        ptr = property(GetPointer, None, doc='''A read only property that returns the underlying script object.''')
        lang = property(GetLanguage, None, doc='''A read only property that returns the script language associated with with this script object.''')
    %}
#endif
}
