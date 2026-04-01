.. _clang_tidy_checks:

LLVM libc clang-tidy checks
===========================

Configuration
-------------

LLVM libc uses layered ``.clang-tidy`` configuration files:

- ``libc/.clang-tidy``: baseline checks for the ``libc`` subtree (currently
  focuses on identifier naming conventions).
- ``libc/src/.clang-tidy``: adds LLVM-libc-specific checks (``llvmlibc-*``) for
  implementation code under ``libc/src`` and also enables
  ``readability-identifier-naming`` and ``llvm-header-guard``. Diagnostics from
  ``llvmlibc-*`` checks are treated as errors.

LLVM-libc checks
----------------

restrict-system-libc-headers
----------------------------
Check name: ``llvmlibc-restrict-system-libc-headers``.

One of libc-project’s design goals is to use kernel headers and compiler
provided headers to prevent code duplication on a per platform basis. This
presents a problem when writing implementations since system libc headers are
easy to include accidentally and we can't just use the ``-nostdinc`` flag.
Improperly included system headers can introduce runtime errors because the C
standard outlines function prototypes and behaviors but doesn’t define
underlying implementation details such as the layout of a struct.

This check prevents accidental inclusion of system libc headers when writing a
libc implementation.

.. code-block:: c++

   #include <stdio.h>            // Not allowed because it is part of system libc.
   #include <stddef.h>           // Allowed because it is provided by the compiler.
   #include "internal/stdio.h"   // Allowed because it is NOT part of system libc.

implementation-in-namespace
---------------------------
Check name: ``llvmlibc-implementation-in-namespace``.

All LLVM-libc implementation constructs must be enclosed in the
``LIBC_NAMESPACE_DECL`` namespace. See :ref:`code_style` for the full technical
rationale and macro definitions.

This check ensures that top-level declarations in a translation unit are
enclosed within the ``LIBC_NAMESPACE_DECL`` namespace.

.. code-block:: c++

    // Correct: implementation inside the correct namespace.
    namespace LIBC_NAMESPACE_DECL {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
        // Namespaces within LIBC_NAMESPACE namespace are allowed.
        namespace inner{
            int localVar = 0;
        }
        // Functions with C linkage are allowed.
        extern "C" void str_fuzz(){}
    }

    // Incorrect: implementation not in a namespace.
    void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}

    // Incorrect: outer most namespace is not correct.
    namespace something_else {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
    }

callee-namespace
----------------
Check name: ``llvmlibc-callee-namespace``.

LLVM-libc is distinct because it is designed to maintain interoperability with
other libc libraries, including the one that lives on the system. This feature
creates some uncertainty about which library a call resolves to especially when
a public header with non-namespaced functions like ``string.h`` is included.

This check ensures any function call resolves to a function within the
LIBC_NAMESPACE namespace.

There are exceptions for the following functions:
``__errno_location`` so that ``errno`` can be set;
``malloc``, ``calloc``, ``realloc``, ``aligned_alloc``, and ``free`` since they
are always external and can be intercepted.

.. code-block:: c++

    namespace LIBC_NAMESPACE_DECL {

    // Allow calls with the fully qualified name.
    LIBC_NAMESPACE::strlen("hello");

    // Allow calls to compiler provided functions.
    (void)__builtin_abs(-1);

    // Bare calls are allowed as long as they resolve to the correct namespace.
    strlen("world");

    // Disallow calling into functions in the global namespace.
    ::strlen("!");

    // Allow calling into specific global functions (explained above)
    ::malloc(10);

    } // namespace LIBC_NAMESPACE_DECL


inline-function-decl
--------------------
Check name: ``llvmlibc-inline-function-decl``.

LLVM libc uses the ``LIBC_INLINE`` macro to tag inline function declarations in
headers. This check enforces that any inline function declaration in a header
begins with ``LIBC_INLINE`` and provides a fix-it to insert the macro.
