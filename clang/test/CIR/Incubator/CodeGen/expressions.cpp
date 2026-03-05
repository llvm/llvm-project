// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

void test(int a) {
// CIR: cir.func {{.*}} @{{.+}}test

  // Should generate LValue parenthesis expression.
  (a) = 1;
  // CIR: %[[CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store{{.*}} %[[CONST]], %{{.+}} : !s32i, !cir.ptr<!s32i>
}