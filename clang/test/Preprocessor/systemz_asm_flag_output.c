// RUN: %clang -target systemz-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s
// RUN: %clang -target s390x-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s

// CHECK: #define __GCC_ASM_FLAG_OUTPUTS__ 1
