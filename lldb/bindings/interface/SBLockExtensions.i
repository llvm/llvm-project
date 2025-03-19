%extend lldb::SBLock {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.Unlock()
    %}
#endif
}
