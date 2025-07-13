%extend lldb::SBMutex {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __enter__(self):
            self.lock()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.unlock()
    %}
#endif
}
