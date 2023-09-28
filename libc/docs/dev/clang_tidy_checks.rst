.. _clang_tidy_checks:

LLVM libc clang-tidy checks
===========================
These are the clang-tidy checks designed to help enforce implementation
standards.
The configuration file is ``src/.clang-tidy``.

restrict-system-libc-header
---------------------------
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

It is part of our implementation standards that all implementation pieces live
under the ``LIBC_NAMESPACE`` namespace. This prevents pollution of the global
namespace. Without a formal check to ensure this, an implementation might
compile and pass unit tests, but not produce a usable libc function.

This check that ensures any function call resolves to a function within the
``LIBC_NAMESPACE`` namespace.

.. code-block:: c++

    // Correct: implementation inside the correct namespace.
    namespace LIBC_NAMESPACE {
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

    namespace LIBC_NAMESPACE {

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

    } // namespace LIBC_NAMESPACE
