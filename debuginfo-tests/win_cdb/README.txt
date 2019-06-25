These are debug info integration tests similar to the ones in the parent
directory, except that these are designed to test compatibility between clang,
lld, and cdb, the command line debugger that ships as part of the Microsoft
Windows SDK. The debugger command language that cdb uses is very different from
gdb and LLDB, so it's useful to be able to write some tests directly in the cdb
command language.
