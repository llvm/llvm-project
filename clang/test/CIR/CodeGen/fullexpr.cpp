// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int go(int const& val);

int go1() {
  auto x = go(1);
  return x;
}

// CHECK: cir.func @_Z3go1v() -> i32 {
// CHECK: %[[#XAddr:]] = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64}
// CHECK: %[[#RVal:]] = cir.scope {
// CHECK-NEXT:   %[[#TmpAddr:]] = cir.alloca i32, cir.ptr <i32>, ["ref.tmp0", init] {alignment = 4 : i64}
// CHECK-NEXT:   %[[#One:]] = cir.cst(1 : i32) : i32
// CHECK-NEXT:   cir.store %[[#One]], %[[#TmpAddr]] : i32, cir.ptr <i32>
// CHECK-NEXT:   %[[#RValTmp:]] = cir.call @_Z2goRKi(%[[#TmpAddr]]) : (!cir.ptr<i32>) -> i32
// CHECK-NEXT:   cir.yield %[[#RValTmp]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: cir.store %[[#RVal]], %[[#XAddr]] : i32, cir.ptr <i32>