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
