.. title:: clang-tidy - llvmlibc-restrict-system-libc-headers

llvmlibc-restrict-system-libc-headers
=====================================

Finds includes of system libc headers not provided by the compiler within
llvm-libc implementations.

.. code-block:: c++

   #include <stdio.h>            // Not allowed because it is part of system libc.
   #include <stddef.h>           // Allowed because it is provided by the compiler.
   #include "internal/stdio.h"   // Allowed because it is NOT part of system libc.


This check is necessary because accidentally including system libc headers can
lead to subtle and hard to detect bugs. For example consider a system libc
whose ``dirent`` struct has slightly different field ordering than llvm-libc.
While this will compile successfully, this can cause issues during runtime
because they are ABI incompatible.

Options
-------

.. option:: Includes

   A string containing a comma separated glob list of allowed include
   filenames. Similar to the -checks glob list for running clang-tidy itself,
   the two wildcard characters are `*` and `-`, to include and exclude globs,
   respectively. The default is `-*`, which disallows all includes.

   This can be used to allow known safe includes such as Linux development
   headers. See :doc:`portability-restrict-system-includes
   <../portability/restrict-system-includes>` for more
   details.
