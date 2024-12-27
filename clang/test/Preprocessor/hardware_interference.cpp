// RUN: %clang_cc1 -E -dM -D__GCC_CONSTRUCTIVE_SIZE=1000 -D__GCC_DESTRUCTIVE_SIZE=1001 %s -verify -Weverything | FileCheck %s
// RUN: %clang_cc1 -D__GCC_CONSTRUCTIVE_SIZE=1000 -D__GCC_DESTRUCTIVE_SIZE=1001 %s -verify -Weverything
// RUN: %clang_cc1 -E -dM -U__GCC_CONSTRUCTIVE_SIZE -U__GCC_DESTRUCTIVE_SIZE %s -verify -Weverything | FileCheck --check-prefix DISABLED %s
// expected-no-diagnostics

// Validate that we can set a new value on the command line without issuing any
// diagnostics and that we can disabled the macro on the command line without
// issuing any diagnostics.

// CHECK: #define __GCC_CONSTRUCTIVE_SIZE 1000
// CHECK: #define __GCC_DESTRUCTIVE_SIZE 1001
// DISABLED-NOT: __GCC_CONSTRUCTIVE_SIZE
// DISABLED-NOT: __GCC_DESTRUCTIVE_SIZE

int main() {
  return 0;
}
