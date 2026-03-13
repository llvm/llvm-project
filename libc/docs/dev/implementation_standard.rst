Convention for implementing entrypoints
=======================================

LLVM-libc entrypoints are defined in the entrypoints document. In this document,
we explain how the entrypoints are implemented. The source layout document
explains that, within the high level ``src`` directory, there exists one
directory for every public header file provided by LLVM-libc. The
implementations of entrypoints live in the directory for the header they belong
to. Some entrypoints are platform specific, and so their implementations are in
a subdirectory with the name of the platform (e.g. ``stdio/linux/remove.cpp``).

Implementation of entrypoints can span multiple ``.cpp`` and ``.h`` files, but
there will be at least one header file with name of the form
``<entrypoint name>.h`` for every entrypoint. This header file is called the
implementation header file. For the ``isalpha`` function, the path to the
implementation header file is ``src/ctype/isalpha.h``.

Implementation Header File Structure
------------------------------------

We will use the ``isalpha`` function from the public ``ctype.h`` header file as an
example. The ``isalpha`` function will be declared in an internal header file
``src/ctype/isalpha.h`` as follows::

    // --- isalpha.h --- //
    #ifndef LLVM_LIBC_SRC_CTYPE_ISALPHA_H
    #define LLVM_LIBC_SRC_CTYPE_ISALPHA_H

    namespace LIBC_NAMESPACE_DECL {

    int isalpha(int c);

    } // namespace LIBC_NAMESPACE_DECL

    #endif LLVM_LIBC_SRC_CTYPE_ISALPHA_H

Notice that the ``isalpha`` function declaration is nested inside the namespace
``LIBC_NAMESPACE_DECL``. All implementation constructs in LLVM-libc are declared
within the namespace ``LIBC_NAMESPACE_DECL``.

``.cpp`` File Structure
-----------------------

The main ``.cpp`` file is named ``<entrypoint name>.cpp`` and is usually in the
same folder as the header. It contains the signature of the entrypoint function,
which must be defined with the ``LLVM_LIBC_FUNCTION`` macro. For example, the
``isalpha`` function from ``ctype.h`` is defined as follows, in the file
``src/ctype/isalpha.cpp``::

    // --- isalpha.cpp --- //

    namespace LIBC_NAMESPACE_DECL {

    LLVM_LIBC_FUNCTION(int, isalpha, (int c)) {
      // ... implementation goes here.
    }

    } // namespace LIBC_NAMESPACE_DECL

Notice the use of the macro ``LLVM_LIBC_FUNCTION``. This macro helps us define
a C alias symbol for the C++ implementation. For example, for a library build,
the macro is defined as follows::

    #define LLVM_LIBC_FUNCTION(type, name, arglist)
        LLVM_LIBC_FUNCTION_IMPL(type, name, arglist)
    #define LLVM_LIBC_FUNCTION_IMPL(type, name, arglist)
        LLVM_LIBC_FUNCTION_ATTR decltype(LIBC_NAMESPACE::name)
            __##name##_impl__ __asm__(#name);
        decltype(LIBC_NAMESPACE::name) name [[gnu::alias(#name)]];
        type __##name##_impl__ arglist

The LLVM_LIBC_FUNCTION_ATTR macro is normally defined to nothing, but can be
defined by vendors who want to set their own attributes.
