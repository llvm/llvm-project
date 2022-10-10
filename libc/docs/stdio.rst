=============================
StdIO Functions in LLVM-libc
=============================

-------
Summary
-------

This document tracks the status of the implementation of stdio functions in LLVM
Libc.

---------------
Source location
---------------

-   The main source for string functions is located at:
    ``libc/src/stdio`` with subdirectories for internal implementations.

---------------------
Implementation Status
---------------------

Formatted Input/Output Functions
================================

These functions take in format strings and arguments of various types and
convert either to or from those arguments. These functions are the current focus
(owner: michaelrj).

=============  =========
Function_Name  Available
=============  =========
\*printf       WIP
\*scanf
=============  =========

``FILE`` Access
===============

These functions are used to interact with the ``FILE`` object type, which is an
I/O stream, often used to represent a file on the host's hard drive. Currently
the ``FILE`` object is only available on linux.

=============  =========
Function_Name  Available
=============  =========
fopen          YES
freopen
fclose         YES
fflush         YES
setbuf
setvbuf
ftell
fgetpos
fseek          YES
fsetpos
rewind
tmpfile
clearerr       YES
feof           YES
ferror         YES
flockfile      YES
funlockfile    YES
=============  =========

Operations on system files
==========================

These functions operate on files on the host's system, without using the 
``FILE`` object type. They only take the name of the file being operated on.

=============  =========
Function_Name  Available
=============  =========
remove         YES
rename
tmpnam
=============  =========

Unformatted ``FILE`` Input/Output Functions
===========================================

The ``gets`` function was removed in C11 for having no bounds checking and
therefor being impossible to use safely.

=============  =========
Function_Name  Available
=============  =========
(f)getc
fgets
getchar
fread          YES
(f)putc
(f)puts        YES
putchar
fwrite         YES
ungetc
=============  =========
