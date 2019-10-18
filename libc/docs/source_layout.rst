LLVM-libc Source Tree Layout
============================

At the top-level, LLVM-libc source tree is organized in to the following
directories::

   + libc
        - cmake
        - docs
        - include
        - lib
        - loader
        - src
        + utils
            - build_scripts
            - testing
        - www

Each of these directories is explained in detail below.

The ``cmake`` directory
-----------------------

The ``cmake`` directory contains the implementations of LLVM-libc's CMake build
rules.

The ``docs`` directory
----------------------

The ``docs`` directory contains design docs and also informative documents like
this document on source layout.

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

1. There is a toplevel CMakeLists.txt file.
2. For every public header file provided by llvm-libc, there exists a
   corresponding directory in the ``src`` directory. The name of the directory
   is same as the base name of the header file. For example, the directory
   corresponding to the public ``math.h`` header file is named ``math``. The
   implementation standard document explains more about the *header*
   directories.

The ``www`` directory
---------------------

The ``www`` directory contains the HTML content of libc.llvm.org

The ``utils/build_scripts`` directory
-------------------------------------

This directory contains scripts which support the build system, tooling etc.

The ``utils/testing`` directory
-------------------------------

This directory contains testing infrastructure.
