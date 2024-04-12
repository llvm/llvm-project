// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int x, short y) {
  return x ?: y;
}

// CHECK: cir.func @foo
// CHECK: %[[Load:.*]] = cir.load
// CHECK: %[[Bool:.*]] = cir.cast(int_to_bool, %[[Load]] : !s32i), !cir.bool loc(#loc8)
// CHECK: = cir.ternary(%[[Bool]], true {
// CHECK:   cir.yield %[[Load]]