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

int f5(int a, int *b, bool c);
int f6() {
  int b = 1;
  return f5(2, &b, false);
}

// CHECK-LABEL: cir.func @_Z2f6v() -> !s32i
// CHECK:         %[[#b:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:         %[[#a:]] = cir.const #cir.int<2> : !s32i
// CHECK-NEXT:    %[[#c:]] = cir.const #false
// CHECK-NEXT:    %{{.+}} = cir.call @_Z2f5iPib(%[[#a]], %[[#b:]], %[[#c]]) : (!s32i, !cir.ptr<!s32i>, !cir.bool) -> !s32i
