.. title:: clang-tidy - llvmlibc-implementation-in-namespace

llvmlibc-implementation-in-namespace
====================================

Checks that all declarations in the llvm-libc implementation are within the
correct namespace.

.. code-block:: c++

    // Implementation inside the LIBC_NAMESPACE namespace.
    // Correct if:
    // - LIBC_NAMESPACE is a macro
    // - LIBC_NAMESPACE expansion starts with `__llvm_libc`
    namespace LIBC_NAMESPACE {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
        // Namespaces within LIBC_NAMESPACE namespace are allowed.
        namespace inner {
            int localVar = 0;
        }
        // Functions with C linkage are allowed.
        extern "C" void str_fuzz() {}
    }

    // Incorrect: implementation not in the LIBC_NAMESPACE namespace.
    void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}

    // Incorrect: outer most namespace is not the LIBC_NAMESPACE macro.
    namespace something_else {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
    }

    // Incorrect: outer most namespace expansion does not start with `__llvm_libc`.
    #define LIBC_NAMESPACE custom_namespace
    namespace LIBC_NAMESPACE {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
    }
