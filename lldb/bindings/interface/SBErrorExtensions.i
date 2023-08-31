STRING_EXTENSION_OUTSIDE(SBError)

%extend lldb::SBError {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            return self.GetError()

        def __hex__(self):
            return self.GetError()

        def __oct__(self):
            return self.GetError()

        def __len__(self):
            pass

        def __iter__(self):
            pass

        value = property(GetError, None, doc='''A read only property that returns the same result as GetError().''')
        fail = property(Fail, None, doc='''A read only property that returns the same result as Fail().''')
        success = property(Success, None, doc='''A read only property that returns the same result as Success().''')
        description = property(GetCString, None, doc='''A read only property that returns the same result as GetCString().''')
        type = property(GetType, None, doc='''A read only property that returns the same result as GetType().''')
    %}
#endif
}
