Convention for implementing entrypoints
=======================================

LLVM-libc entrypoints are defined in the entrypoints document. In this document,
we explain how the entrypoints are implemented. The source layout document
explains that, within the high level ``src`` directory, there exists one
directory for every public header file provided by LLVM-libc. The
implementations of related group of entrypoints will also live in a directory of
their own. This directory will have a name indicative of the related group of
entrypoints, and will be under the directory corresponding to the header file of
the entrypoints. For example, functions like ``fopen`` and ``fclose`` cannot be
tested independent of each other and hence will live in a directory named
``src/stdio/file_operations``. On the other hand, the implementation of the
``round`` function from ``math.h`` can be tested by itself, so it will live in
the directory of its own named ``src/math/round/``.

Implementation of entrypoints can span multiple ``.cpp`` and ``.h`` files, but
there will be at least one header file with name of the form
``<entrypoint name>.h`` for every entrypoint. This header file is called as the
implementation header file. For the ``round`` function, the path to the
implementation header file will be ``src/math/round/round.h``. The rest of this
document explains the structure of implementation header files and ``.cpp``
files.

Implementation Header File Structure
------------------------------------

We will use the ``round`` function from the public ``math.h`` header file as an
example. The ``round`` function will be declared in an internal header file
``src/math/round/round.h`` as follows::

    // --- round.h --- //
    #ifndef LLVM_LIBC_SRC_MATH_ROUND_ROUND_H
    #define LLVM_LIBC_SRC_MATH_ROUND_ROUND_H

    namespace LIBC_NAMESPACE {

    double round(double);

    } // namespace LIBC_NAMESPACE

    #endif LLVM_LIBC_SRC_MATH_ROUND_ROUND_H

Notice that the ``round`` function declaration is nested inside the namespace
``LIBC_NAMESPACE``. All implementation constructs in LLVM-libc are declared within
the namespace ``LIBC_NAMESPACE``.

``.cpp`` File Structure
-----------------------

The implementation can span multiple ``.cpp`` files. However, the signature of
the entrypoint function should make use of a special macro. For example, the
``round`` function from ``math.h`` should be defined as follows, say in the file
``src/math/math/round.cpp``::

    // --- round.cpp --- //

    namespace LIBC_NAMESPACE {

    double LLVM_LIBC_ENTRYPOINT(round)(double d) {
      // ... implementation goes here.
    }

    } // namespace LIBC_NAMESPACE

Notice the use of the macro ``LLVM_LIBC_ENTRYPOINT``. This macro helps us define
an C alias symbol for the C++ implementation. The C alias need not be added by
the macro by itself. For example, for ELF targets, the macro is defined as
follows::

    #define ENTRYPOINT_SECTION_ATTRIBUTE(name) \
        __attribute__((section(".llvm.libc.entrypoint."#name)))
    #define LLVM_LIBC_ENTRYPOINT(name) ENTRYPOINT_SECTION_ATTRIBUTE(name) name

The macro places the C++ function in a unique section with name
``.llvm.libc.entrypoint.<function name>``. This allows us to add a C alias using
a post build step. For example, for the ``round`` function, one can use
``objcopy`` to add an alias symbol as follows::

    objcopy --add-symbol round=.llvm.libc.entrypoint.round:0,function round.o

NOTE: We use a post build ``objcopy`` step to add an alias instead of using
the ``__attribute__((alias))``. For C++, this ``alias`` attribute requires
mangled names of the referees. Using the post build ``objcopy`` step helps
us avoid putting mangled names with ``alias`` attributes.
