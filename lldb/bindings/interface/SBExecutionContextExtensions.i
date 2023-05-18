%extend lldb::SBExecutionContext {
#ifdef SWIGPYTHON
    %pythoncode %{
        target = property(GetTarget, None, doc='''A read only property that returns the same result as GetTarget().''')
        process = property(GetProcess, None, doc='''A read only property that returns the same result as GetProcess().''')
        thread = property(GetThread, None, doc='''A read only property that returns the same result as GetThread().''')
        frame = property(GetFrame, None, doc='''A read only property that returns the same result as GetFrame().''')
    %}
#endif
}
