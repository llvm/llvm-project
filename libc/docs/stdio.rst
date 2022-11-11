===============
StdIO Functions
===============

.. include:: check.rst

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
fopen          |check|
freopen
fclose         |check|
fflush         |check|
setbuf
setvbuf
ftell
fgetpos
fseek          |check|
fsetpos
rewind
tmpfile
clearerr       |check|
feof           |check|
ferror         |check|
flockfile      |check|
funlockfile    |check|
=============  =========

Operations on system files
==========================

These functions operate on files on the host's system, without using the 
``FILE`` object type. They only take the name of the file being operated on.

=============  =========
Function_Name  Available
=============  =========
remove         |check|
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
fread          |check|
(f)putc
(f)puts        |check|
putchar
fwrite         |check|
ungetc
=============  =========
