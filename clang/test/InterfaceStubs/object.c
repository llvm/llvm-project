// RUN: %clang_cc1 -fvisibility default -o - -emit-interface-stubs %s | FileCheck -check-prefix=CHECK-TAPI %s
// RUN: %clang -fvisibility=default -c -o %t.o %s
// RUN: llvm-nm %t.o 2>&1 | FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK-TAPI: data", Type: Object, Size: 4 }
// CHECK-SYMBOLS: data
int data = 42;
