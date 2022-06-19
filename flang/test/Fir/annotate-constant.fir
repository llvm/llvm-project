// RUN: fir-opt -annotate-constant %s | FileCheck %s

// CHECK-LABEL: func @annotate_test() -> !fir.ref<!fir.array<?xi32>> {
func.func @annotate_test() -> !fir.ref<!fir.array<?xi32>> {
  %1 = arith.constant 5 : index
  // CHECK: %[[a:.*]] = fir.alloca !fir.array<?xi32>, %{{.*}} {canonicalize_constant_operands = [5 : index]}
  %2 = fir.alloca !fir.array<?xi32>, %1
  return %2 : !fir.ref<!fir.array<?xi32>>
}
