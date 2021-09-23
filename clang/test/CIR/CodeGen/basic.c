// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -fcir-warnings %s -fcir-output=%t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

int foo(int i) {
  return i;
}

// CHECK: module  {
// CHECK-NEXT:   func @foo(%arg0: i32) -> i32 {
// CHECK-NEXT:     %0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %0[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %0[] : memref<i32>
// CHECK-NEXT:     cir.return %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
