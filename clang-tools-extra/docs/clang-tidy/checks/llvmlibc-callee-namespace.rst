.. title:: clang-tidy - llvmlibc-callee-namespace

llvmlibc-callee-namespace
====================================

Checks all calls resolve to functions within ``__llvm_libc`` namespace.

.. code-block:: c++
    
    namespace __llvm_libc {

    // Allow calls with the fully qualified name.
    __llvm_libc::strlen("hello");

    // Allow calls to compiler provided functions.
    (void)__builtin_abs(-1);

    // Bare calls are allowed as long as they resolve to the correct namespace.
    strlen("world");

    // Disallow calling into functions in the global namespace.
    ::strlen("!");

    } // namespace __llvm_libc
