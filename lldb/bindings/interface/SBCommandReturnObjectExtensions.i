STRING_EXTENSION_OUTSIDE(SBCommandReturnObject)

%extend lldb::SBCommandReturnObject {
#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
    def __eq__(self, other):
      return not self.__ne__(other)

    def __int__(self):
      pass

    def __len__(self):
      pass

    def __hex__(self):
      pass

    def __oct__(self):
      pass

    def __iter__(self):
      pass
    %}
#endif

    // transfer_ownership does nothing, and is here for compatibility with
    // old scripts.  Ownership is tracked by reference count in the ordinary way.

    void SetImmediateOutputFile(lldb::FileSP BORROWED, bool transfer_ownership) {
        self->SetImmediateOutputFile(BORROWED);
    }
    void SetImmediateErrorFile(lldb::FileSP BORROWED, bool transfer_ownership) {
        self->SetImmediateErrorFile(BORROWED);
    }

    // wrapping the variadic Printf() with a plain Print()
    // because it is hard to support varargs in SWIG bridgings
    void Print (const char* str)
    {
        self->Printf("%s", str);
    }
}
