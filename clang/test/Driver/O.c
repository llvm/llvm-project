// Test that we parse and translate the -O option correctly.

// RUN: %clang -O -### %s 2>&1 | FileCheck -check-prefix=CHECK-O %s
// CHECK-O: -O1

// RUN: %clang -O0 -### %s 2>&1 | FileCheck -check-prefix=CHECK-O0 %s
// CHECK-O0: -O0

// RUN: %clang -O1 -### %s 2>&1 | FileCheck -check-prefix=CHECK-O1 %s
// CHECK-O1: -O1
