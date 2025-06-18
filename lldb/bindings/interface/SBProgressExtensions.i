%extend lldb::SBProgress {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __enter__(self):
            '''No-op for with statement'''
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            '''Finalize the progress object'''
            self.Finalize()
    %}
#endif
}