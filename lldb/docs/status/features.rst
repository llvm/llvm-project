Features
========

LLDB supports a broad variety of basic debugging features such as reading DWARF, supporting step, next, finish, backtraces, etc. Some more interested bits are:

* Plug-in architecture for portability and extensibility:

  * Object file parsers for executable file formats. Support currently includes Mach-O (32 and 64-bit) & ELF (32-bit).
  * Object container parsers to extract object files contained within a file. Support currently includes universal Mach-O files & BSD Archives.
  * Debug symbol file parsers to incrementally extract debug information from object files. Support currently includes DWARF & Mach-O symbol tables.
  * Symbol vendor plug-ins collect data from a variety of different sources for an executable object.
  * Disassembly plug-ins for each architecture. Support currently includes an LLVM disassembler for i386, x86-64 , ARM/Thumb, and PPC64le
  * Debugger plug-ins implement the host and target specific functions required to debug.

* SWIG-generated script bridging allows Python to access and control the public API of the debugger library.
* A remote protocol server, debugserver, implements Mac OS X debugging on i386 and x86-64.
* A command line debugger - the lldb executable itself.
* A framework API to the library.
