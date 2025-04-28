// RUN: fir-opt --cse -split-input-file %s | FileCheck %s

// Check that the redundant fir.load is removed.
func.func @fun(%arg0: !fir.ref<i64>) -> i64 {
    %0 = fir.load %arg0 : !fir.ref<i64>
    %1 = fir.load %arg0 : !fir.ref<i64>
    %2 = arith.addi %0, %1 : i64
    return %2 : i64
}

// CHECK-LABEL: func @fun
// CHECK-NEXT:    %[[LOAD:.*]] = fir.load %{{.*}} : !fir.ref<i64>
// CHECK-NEXT:    %{{.*}} = arith.addi %[[LOAD]], %[[LOAD]] : i64

// -----

// CHECK-LABEL: func @fun(
// CHECK-SAME:            %[[A:.*]]: !fir.ref<i64>
func.func @fun(%a : !fir.ref<i64>) -> i64 {
  // CHECK: %[[LOAD:.*]] = fir.load %[[A]] : !fir.ref<i64>
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi %[[LOAD]], %[[LOAD]] : i64
  %3 = arith.addi %1, %2 : i64
  %4 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %5 = arith.addi %3, %4 : i64
  %6 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %7 = arith.addi %5, %6 : i64
  %8 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %9 = arith.addi %7, %8 : i64
  %10 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %11 = arith.addi %10, %9 : i64
  %12 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %13 = arith.addi %11, %12 : i64
  // CHECK-NEXT: return %{{.*}} : i64
  return %13 : i64
}

// -----

func.func @fun(%a : !fir.ref<i64>) -> i64 {
  cf.br ^bb1
^bb1:
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  %3 = arith.addi %1, %2 : i64
  cf.br ^bb2
^bb2:
  %4 = fir.load %a : !fir.ref<i64>
  %5 = arith.subi %4, %4 : i64
  return %5 : i64
}
