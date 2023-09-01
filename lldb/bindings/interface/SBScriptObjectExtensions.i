%extend lldb::SBScriptObject {
#ifdef SWIGPYTHON
    %pythoncode %{
        ptr = property(GetPointer, None, doc='''A read only property that returns the underlying script object.''')
        lang = property(GetLanguage, None, doc='''A read only property that returns the script language associated with with this script object.''')
    %}
#endif
}
