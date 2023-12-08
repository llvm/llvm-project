%feature("docstring",
"Represents a file."
) lldb::SBFile;

%feature("docstring", "
Initialize a SBFile from a file descriptor.  mode is
'r', 'r+', or 'w', like fdopen.") lldb::SBFile::SBFile;

%feature("docstring", "initialize a SBFile from a python file object") lldb::SBFile::SBFile;

%feature("autodoc", "Read(buffer) -> SBError, bytes_read") lldb::SBFile::Read;
%feature("autodoc", "Write(buffer) -> SBError, written_read") lldb::SBFile::Write;

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
    file object using ``open(fd, mode=..., closefd=False)``
") lldb::SBFile::GetFile;
