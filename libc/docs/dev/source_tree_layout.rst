.. _source_tree_layout:

============================
LLVM-libc Source Tree Layout
============================

At the top-level, LLVM-libc source tree is organized in to the following
directories::

   + libc
        - benchmarks
        - cmake
        - config
        - docs
        - examples
        - fuzzing
        - hdr
        - include
        - lib
        - src
        - startup
        - test
        - utils

Each of these directories is explained breifly below.

The ``benchmarks`` directory
----------------------------

The ``benchmarks`` directory contains LLVM-libc's benchmarking utilities. These
are mostly used for the memory functions.

The ``config`` directory
------------------------

The ``config`` directory contains the default configurations for the targets
LLVM-libc supports. These are files in the ``config/<platform>/<architecture>/``
subdirectory called ``entrypoints.txt``, ``exclude.txt``, ``headers.txt``,  and
``config.json``. These tell cmake which entrypoints are available, which
entrypoints to exclude, which headers to generate, and what options to set for
the current target respectively. There are also other platform specific files in
the ``config/<platform>/`` subdirectory.

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

This directory contains fuzzing tests for the various components of LLVM-libc.
The directory structure within this directory mirrors the directory structure
of the top-level ``libc`` directory itself. For more details, see
:doc:`fuzzing`.

The ``hdr`` directory
---------------------

This directory contains proxy headers which are included from the files in the
src directory. These proxy headers either include our internal type or macro
definitions, or the system's type or macro definitions, depending on if we are
in fullbuild or overlay mode.

The ``include`` directory
-------------------------

The ``include`` directory contains:

1. ``*.h.def`` files - These files are used to construct the generated public
   header files.
2. Self contained public header files - These are header files which are
   already in the form that get installed when LLVM-libc is installed on a
   user's computer. These are mostly in the ``llvm-libc-macros`` and
   ``llvm-libc-types`` subdirectories.

The ``lib`` directory
---------------------

This directory contains a ``CMakeLists.txt`` file listing the targets for the
public libraries ``libc.a``, ``libm.a`` etc.

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

The ``startup`` directory
-------------------------

This directory contains the implementations of the application startup objects
like ``crt1.o`` etc.

The ``test`` directory
----------------------

This directory contains tests for the various components of LLVM-libc. The
directory structure within this directory mirrors the directory structure of the
toplevel ``libc`` directory itself. A test for, say the ``mmap`` function, lives
in the directory ``test/src/sys/mman/`` as implementation of ``mmap`` lives in
``src/sys/mman``.

The ``utils`` directory
-----------------------

This directory contains utilities used by other parts of the LLVM-libc system.
See the `README` files in the subdirectories within this directory to learn
about the various utilities.
