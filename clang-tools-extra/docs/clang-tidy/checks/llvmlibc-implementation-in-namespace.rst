.. title:: clang-tidy - llvmlibc-implementation-in-namespace

llvmlibc-implementation-in-namespace
====================================

Checks that all declarations in the llvm-libc implementation are within the
correct namespace.

.. code-block:: c++

    // Correct: implementation inside the correct namespace.
    namespace __llvm_libc {
        void LLVM_LIBC_ENTRYPOINT(strcpy)(char *dest, const char *src) {}
        // Namespaces within __llvm_libc namespace are allowed.
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
