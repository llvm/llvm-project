.. title:: clang-tidy - llvmlibc-implementation-in-namespace

llvmlibc-implementation-in-namespace
====================================

Checks that all declarations in the llvm-libc implementation are within the
correct namespace.

.. code-block:: c++

    // Implementation inside the LIBC_NAMESPACE_DECL namespace.
    // Correct if:
    // - LIBC_NAMESPACE_DECL is a macro
    // - LIBC_NAMESPACE_DECL expansion starts with `[[gnu::visibility("hidden")]] __llvm_libc`
    namespace LIBC_NAMESPACE_DECL {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
        // Namespaces within LIBC_NAMESPACE_DECL namespace are allowed.
        namespace inner {
            int localVar = 0;
        }
        // Functions with C linkage are allowed.
        extern "C" void str_fuzz() {}
    }

    // Incorrect: implementation not in the LIBC_NAMESPACE_DECL namespace.
    void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}

    // Incorrect: outer most namespace is not the LIBC_NAMESPACE_DECL macro.
    namespace something_else {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
    }

    // Incorrect: outer most namespace expansion does not start with `[[gnu::visibility("hidden")]] __llvm_libc`.
    #define LIBC_NAMESPACE_DECL custom_namespace
    namespace LIBC_NAMESPACE_DECL {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
    }
