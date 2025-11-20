.. title:: clang-tidy - llvmlibc-callee-namespace

llvmlibc-callee-namespace
====================================

Checks all calls resolve to functions within correct namespace.

.. code-block:: c++

    // Implementation inside the LIBC_NAMESPACE namespace.
    // Correct if:
    // - LIBC_NAMESPACE is a macro
    // - LIBC_NAMESPACE expansion starts with `__llvm_libc`
    namespace LIBC_NAMESPACE {

    // Allow calls with the fully qualified name.
    LIBC_NAMESPACE::strlen("hello");

    // Allow calls to compiler provided functions.
    (void)__builtin_abs(-1);

    // Bare calls are allowed as long as they resolve to the correct namespace.
    strlen("world");

    // Disallow calling into functions in the global namespace.
    ::strlen("!");

    } // namespace LIBC_NAMESPACE
