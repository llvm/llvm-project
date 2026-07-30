/// GCC --disable-multiarch, GCC --enable-multiarch (upstream and Debian specific) have different behaviors.
/// We choose not to support the option.

// RUN: not %clang -print-multiarch --target=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s

// CHECK: error: unsupported option '-print-multiarch'
