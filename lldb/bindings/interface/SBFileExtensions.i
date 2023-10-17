%extend lldb::SBFile {
    static lldb::SBFile MakeBorrowed(lldb::FileSP BORROWED) {
        return lldb::SBFile(BORROWED);
    }
    static lldb::SBFile MakeForcingIOMethods(lldb::FileSP FORCE_IO_METHODS) {
        return lldb::SBFile(FORCE_IO_METHODS);
    }
    static lldb::SBFile MakeBorrowedForcingIOMethods(lldb::FileSP BORROWED_FORCE_IO_METHODS) {
        return lldb::SBFile(BORROWED_FORCE_IO_METHODS);
    }

#ifdef SWIGPYTHON
    %pythoncode {
        @classmethod
        def Create(cls, file, borrow=False, force_io_methods=False):
            """
            Create a SBFile from a python file object, with options.

            If borrow is set then the underlying file will
            not be closed when the SBFile is closed or destroyed.

            If force_scripting_io is set then the python read/write
            methods will be called even if a file descriptor is available.
            """
            if borrow:
                if force_io_methods:
                    return cls.MakeBorrowedForcingIOMethods(file)
                else:
                    return cls.MakeBorrowed(file)
            else:
                if force_io_methods:
                    return cls.MakeForcingIOMethods(file)
                else:
                    return cls(file)
    }
#endif
}
