// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -fcir-warnings %s -fcir-output=%t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

int foo(int i) {
  return i;
}

//      CHECK: module  {
// CHECK-NEXT:   func @foo(%arg0: i32) -> i32 {
// CHECK-NEXT:     cir.return %arg0 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }
