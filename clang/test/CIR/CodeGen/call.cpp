// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

void f1() {}
void f2() {
  f1();
}

// CHECK-LABEL: cir.func @_Z2f1v
// CHECK-LABEL: cir.func @_Z2f2v
// CHECK:         cir.call @_Z2f1v() : () -> ()

int f3() { return 2; }
int f4() {
  int x = f3();
  return x;
}

// CHECK-LABEL: cir.func @_Z2f3v() -> !s32i
// CHECK-LABEL: cir.func @_Z2f4v() -> !s32i
// CHECK:         cir.call @_Z2f3v() : () -> !s32i
