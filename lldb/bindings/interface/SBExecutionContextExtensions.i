%extend lldb::SBExecutionContext {
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

        target = property(GetTarget, None, doc='''A read only property that returns the same result as GetTarget().''')
        process = property(GetProcess, None, doc='''A read only property that returns the same result as GetProcess().''')
        thread = property(GetThread, None, doc='''A read only property that returns the same result as GetThread().''')
        frame = property(GetFrame, None, doc='''A read only property that returns the same result as GetFrame().''')
    %}
#endif
}
