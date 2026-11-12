// RUN: %clang -target aarch64-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s

// CHECK: #define __GCC_ASM_FLAG_OUTPUTS__ 1
