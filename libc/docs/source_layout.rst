LLVM-libc Source Tree Layout
============================

At the top-level, LLVM-libc source tree is organized in to the following
directories::

   + libc
        - cmake
        - docs
        - fuzzing
        - include
        - lib
        - loader
        - src
        - test
        - utils

Each of these directories is explained in detail below.

The ``cmake`` directory
-----------------------

The ``cmake`` directory contains the implementations of LLVM-libc's CMake build
rules.

The ``docs`` directory
----------------------

The ``docs`` directory contains design docs and also informative documents like
this document on source layout.

The ``fuzzing`` directory
-------------------------

This directory contains fuzzing tests for the various components of llvm-libc. The
directory structure within this directory mirrors the directory structure of the
top-level ``libc`` directory itself. For more details, see :doc:`fuzzing`.

The ``include`` directory
-------------------------

The ``include`` directory contains:

1. Self contained public header files - These are header files which are
   already in the form that get installed when LLVM-libc is installed on a user's
   computer.
2. ``*.h.def`` and ``*.h.in`` files - These files are used to construct the
   generated public header files.
3. A ``CMakeLists.txt`` file - This file lists the targets for the self
   contained and generated public header files.

The ``lib`` directory
---------------------

This directory contains a ``CMakeLists.txt`` file listing the targets for the
public libraries ``libc.a``, ``libm.a`` etc.

The ``loader`` directory
------------------------

This directory contains the implementations of the application loaders like
``crt1.o`` etc.

The ``src`` directory
---------------------

This directory contains the implementations of the llvm-libc entrypoints. It is
further organized as follows:

1. There is a top-level CMakeLists.txt file.
2. For every public header file provided by llvm-libc, there exists a
   corresponding directory in the ``src`` directory. The name of the directory
   is same as the base name of the header file. For example, the directory
   corresponding to the public ``math.h`` header file is named ``math``. The
   implementation standard document explains more about the *header*
   directories.

The ``test`` directory
----------------------

This directory contains tests for the various components of llvm-libc. The
directory structure within this directory mirrors the directory structure of the
toplevel ``libc`` directory itself. A test for, say the ``mmap`` function, lives
in the directory ``test/src/sys/mman/`` as implementation of ``mmap`` lives in
``src/sys/mman``.

The ``utils`` directory
-----------------------

This directory contains utilities used by other parts of the llvm-libc system.
See the `README` files, in the sub-directories within this directory, to learn
about the various utilities.
