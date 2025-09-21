// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void conditionalResultIimplicitCast(int a, int b, float f) {
  // Should implicit cast back to int.
  int x = a && b;
  // CHECK: %[[#INT:]] = cir.ternary
  // CHECK: %{{.+}} = cir.cast(bool_to_int, %[[#INT]] : !cir.bool), !s32i
  float y = f && f;
  // CHECK: %[[#BOOL:]] = cir.ternary
  // CHECK: %[[#INT:]] = cir.cast(bool_to_int, %[[#BOOL]] : !cir.bool), !s32i
  // CHECK: %{{.+}} = cir.cast(int_to_float, %[[#INT]] : !s32i), !cir.float
}
