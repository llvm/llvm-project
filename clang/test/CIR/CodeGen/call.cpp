// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

void f1();
void f2() {
  f1();
}

// CHECK-LABEL: cir.func @f2
// CHECK:         cir.call @f1() : () -> ()

int f3();
int f4() {
  int x = f3();
  return x;
}

// CHECK-LABEL: cir.func @f4() -> !s32i
// CHECK:         %[[#x:]] = cir.call @f3() : () -> !s32i
// CHECK-NEXT:    cir.store %[[#x]], %{{.+}} : !s32i, !cir.ptr<!s32i>
