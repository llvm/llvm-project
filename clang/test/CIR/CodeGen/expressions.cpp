// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void test(int a) {
// CHECK: cir.func @{{.+}}test

  // Should generate LValue parenthesis expression.
  (a) = 1;
  // CHECK: %[[#C:]] = cir.const(#cir.int<1> : !s32i) : !s32i
  // CHECK: cir.store %[[#C]], %{{.+}} : !s32i, cir.ptr <!s32i>
}
