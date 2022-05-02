// Check that clang reports an error message if -flto without -c is used
// on a toolchain that is not expecting it (HasNativeLLVMSupport() is false).

// RUN: %clang -### -flto -target x86_64-unknown-unknown %s 2>&1 | FileCheck %s
// CHECK: error: {{.*}} unable to pass LLVM bit-code files to linker
