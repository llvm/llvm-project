// Check that clang reports an error message if -flto without -c is used
// on a toolchain that is not expecting it (HasNativeLLVMSupport() is false).

// RUN: not %clang -### -flto --target=x86_64-unknown-unknown %s 2>&1 | FileCheck %s
// CHECK: error: {{.*}} unable to pass LLVM bit-code files to linker

// RUN: %clang -### -flto --target=arm-none-eabi %s 2>&1 | FileCheck /dev/null --implicit-check-not=error:
