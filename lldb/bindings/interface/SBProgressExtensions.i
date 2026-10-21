%extend lldb::SBProgress {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __enter__(self):
            '''Returns the current object'''
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            '''Finalize the progress object'''
            self.Finalize()
    %}
#endif
}