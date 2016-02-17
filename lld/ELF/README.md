The New ELF Linker
==================
This directory contains a port of the new PE/COFF linker for ELF.

Overall Design
--------------
See COFF/README.md for details on the design. Note that unlike COFF, we do not
distinguish chunks from input sections; they are merged together.

Capabilities
------------
This linker can link LLVM and Clang on Linux/x86-64 or FreeBSD/x86-64
"Hello world" can be linked on Linux/PPC64 and on Linux/AArch64 or
FreeBSD/AArch64.

Performance
-----------
Achieving good performance is one of our goals. It's too early to reach a
conclusion, but we are optimistic about that as it currently seems to be faster
than GNU gold. It will be interesting to compare when we are close to feature
parity.

Library Use
-----------

You can embed LLD to your program by linking against it and calling the linker's
entry point function lld::elf2::link.

The current policy is that it is your reponsibility to give trustworthy object
files. The function is guaranteed to return as long as you do not pass corrupted
or malicious object files. A corrupted file could cause a fatal error or SEGV.
That being said, you don't need to worry too much about it if you create object
files in a usual way and give them to the linker (it is naturally expected to
work, or otherwise it's a linker's bug.)
