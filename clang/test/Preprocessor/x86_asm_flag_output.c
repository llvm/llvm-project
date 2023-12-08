// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s
// RUN: %clang -target x86_64-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s

// CHECK: #define __GCC_ASM_FLAG_OUTPUTS__ 1
