// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int go(int const& val);

int go1() {
  auto x = go(1);
  return x;
}

// CHECK: cir.func @_Z3go1v() -> !s32i
// CHECK: %[[#XAddr:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK: %[[#RVal:]] = cir.scope {
// CHECK-NEXT:   %[[#TmpAddr:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["ref.tmp0", init] {alignment = 4 : i64}
// CHECK-NEXT:   %[[#One:]] = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:   cir.store %[[#One]], %[[#TmpAddr]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %[[#RValTmp:]] = cir.call @_Z2goRKi(%[[#TmpAddr]]) : (!cir.ptr<!s32i>) -> !s32i
// CHECK-NEXT:   cir.yield %[[#RValTmp]] : !s32i
// CHECK-NEXT: }
// CHECK-NEXT: cir.store %[[#RVal]], %[[#XAddr]] : !s32i, cir.ptr <!s32i>
