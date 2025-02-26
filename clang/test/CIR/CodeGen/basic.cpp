// RUN: not %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

// This error is caused by the "const int i = 2" line in f2(). When
// initaliziers are implemented, the checks there should be updated
// and the "not" should be removed from the run line.
// CHECK: error: ClangIR code gen Not Yet Implemented: emitAutoVarInit

int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @f1() -> !cir.int<s, 32>
// CHECK:    %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i"] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:    cir.return %[[I]] : !cir.int<s, 32>

int f2() {
  const int i = 2;
  return i;
}

// CHECK: cir.func @f2() -> !cir.int<s, 32>
// CHECK:    %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", const] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CHECK:    cir.return %[[I]] : !cir.int<s, 32>
