// RUN: %clang -c -o - -emit-interface-stubs %s | FileCheck -check-prefix=CHECK-TAPI %s
// RUN: %clang -c -o - -emit-interface-stubs %s | FileCheck -check-prefix=CHECK-SYMBOLS %s
// RUN: %clang -c -o - %s | llvm-nm - 2>&1 | FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK-TAPI: data: { Type: Object, Size: 4 }
// CHECK-SYMBOLS: data
int data = 42;
