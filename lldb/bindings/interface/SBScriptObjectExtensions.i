STRING_EXTENSION_OUTSIDE(SBScriptObject)

%extend lldb::SBScriptObject {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

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

        ptr = property(GetPointer, None, doc='''A read only property that returns the underlying script object.''')
        lang = property(GetLanguage, None, doc='''A read only property that returns the script language associated with with this script object.''')
    %}
#endif
}
