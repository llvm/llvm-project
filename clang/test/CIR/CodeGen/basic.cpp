// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int f1();
int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @f1() -> !cir.int<s, 32>
// CHECK:    %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i"] {alignment = 4 : i64}
// CIR-NEXT: %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CIR-NEXT: cir.return %[[I]] : !cir.int<s, 32>
