// RUN: fir-opt %s --fir-to-scf | FileCheck %s

// CHECK-LABEL:   func.func @test_only(
// CHECK-SAME:      %[[ARG0:.*]]: i1, %[[ARG1:.*]]: i32) {
// CHECK:           scf.if %[[ARG0]] {
// CHECK:             %[[VAL_1:.*]] = arith.addi %[[ARG1]], %[[ARG1]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_only(%arg0 : i1, %arg1 : i32) {
  fir.if %arg0 {
    %0 = arith.addi %arg1, %arg1 : i32
  }
  return
}

// CHECK-LABEL:   func.func @test_else()   {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           scf.if %[[VAL_1]] {
// CHECK:             %[[VAL_3:.*]] = arith.constant 3 : i32
// CHECK:           } else {
// CHECK:             %[[VAL_3:.*]] = arith.constant 3 : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_else()   {
  %false = arith.constant false
  %1 = arith.constant 2 : i32
  fir.if %false {
    %2 = arith.constant 3 : i32
  } else {
    %3 = arith.constant 3 : i32
  }
  return
}

// CHECK-LABEL:   func.func @test_two_result()  {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[RES:[0-9]+]]:2 = scf.if %[[VAL_2]] -> (f32, f32) {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_1]] : f32, f32
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_1]] : f32, f32
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test_two_result() {
  %1 = arith.constant 2.0 : f32
  %cmp = arith.constant false
  %x, %y = fir.if %cmp -> (f32, f32) {
    fir.result %1, %1 : f32, f32
  } else {
    fir.result %1, %1 : f32, f32
  }
  return
}
