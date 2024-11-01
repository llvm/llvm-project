STRING_EXTENSION_OUTSIDE(SBError)

%extend lldb::SBError {
#ifdef SWIGPYTHON
    %pythoncode %{
        value = property(GetError, None, doc='''A read only property that returns the same result as GetError().''')
        fail = property(Fail, None, doc='''A read only property that returns the same result as Fail().''')
        success = property(Success, None, doc='''A read only property that returns the same result as Success().''')
        description = property(GetCString, None, doc='''A read only property that returns the same result as GetCString().''')
        type = property(GetType, None, doc='''A read only property that returns the same result as GetType().''')
    %}
#endif
}
