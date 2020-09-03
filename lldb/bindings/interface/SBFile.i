//===-- SWIG Interface for SBFile -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a file."
) SBFile;

class SBFile
{
public:

    SBFile();

    %feature("docstring", "
    Initialize a SBFile from a file descriptor.  mode is
    'r', 'r+', or 'w', like fdopen.");
    SBFile(int fd, const char *mode, bool transfer_ownership);

    %feature("docstring", "initialize a SBFile from a python file object");
    SBFile(FileSP file);

    %extend {
        static lldb::SBFile MakeBorrowed(lldb::FileSP BORROWED) {
            return lldb::SBFile(BORROWED);
        }
        static lldb::SBFile MakeForcingIOMethods(lldb::FileSP FORCE_IO_METHODS) {
            return lldb::SBFile(FORCE_IO_METHODS);
        }
        static lldb::SBFile MakeBorrowedForcingIOMethods(lldb::FileSP BORROWED_FORCE_IO_METHODS) {
            return lldb::SBFile(BORROWED_FORCE_IO_METHODS);
        }
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

    ~SBFile ();

    %feature("autodoc", "Read(buffer) -> SBError, bytes_read") Read;
    SBError Read(uint8_t *buf, size_t num_bytes, size_t *OUTPUT);

    %feature("autodoc", "Write(buffer) -> SBError, written_read") Write;
    SBError Write(const uint8_t *buf, size_t num_bytes, size_t *OUTPUT);

    void Flush();

    bool IsValid() const;

    operator bool() const;

    SBError Close();

    %feature("docstring", "
    Convert this SBFile into a python io.IOBase file object.

    If the SBFile is itself a wrapper around a python file object,
    this will return that original object.

    The file returned from here should be considered borrowed,
    in the sense that you may read and write to it, and flush it,
    etc, but you should not close it.   If you want to close the
    SBFile, call SBFile.Close().

    If there is no underlying python file to unwrap, GetFile will
    use the file descriptor, if available to create a new python
    file object using `open(fd, mode=..., closefd=False)`
    ");
    FileSP GetFile();
};

} // namespace lldb
