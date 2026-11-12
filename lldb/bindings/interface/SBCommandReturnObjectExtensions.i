STRING_EXTENSION_OUTSIDE(SBCommandReturnObject)

%extend lldb::SBCommandReturnObject {
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
