// RUN: fir-opt %s --allow-unregistered-dialect --mem2reg --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @basic() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : i32
// CHECK:           return %[[CONSTANT_0]] : i32
// CHECK:         }
func.func @basic() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = fir.alloca i32
  fir.store %0 to %1 : !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  return %2 : i32
}

// -----

// CHECK-LABEL:   func.func @default_value() -> i32 {
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : i32
// CHECK:           return %[[UNDEFINED_0]] : i32
// CHECK:         }
func.func @default_value() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = fir.alloca i32
  %2 = fir.load %1 : !fir.ref<i32>
  fir.store %0 to %1 : !fir.ref<i32>
  return %2 : i32
}

// -----

// CHECK-LABEL:   func.func @basic_float() -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5.200000e+00 : f32
// CHECK:           return %[[CONSTANT_0]] : f32
// CHECK:         }
func.func @basic_float() -> f32 {
  %0 = arith.constant 5.2 : f32
  %1 = fir.alloca f32
  fir.store %0 to %1 : !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  return %2 : f32
}

// -----

// CHECK-LABEL:   func.func @cycle(
// CHECK-SAME:                     %[[ARG0:.*]]: i64,
// CHECK-SAME:                     %[[ARG1:.*]]: i1,
// CHECK-SAME:                     %[[ARG2:.*]]: i64) {
// CHECK:           cf.cond_br %[[ARG1]], ^bb1(%[[ARG2]] : i64), ^bb2(%[[ARG2]] : i64)
// CHECK:         ^bb1(%[[VAL_0:.*]]: i64):
// CHECK:           "test.use"(%[[VAL_0]]) : (i64) -> ()
// CHECK:           cf.br ^bb2(%[[ARG0]] : i64)
// CHECK:         ^bb2(%[[VAL_1:.*]]: i64):
// CHECK:           cf.br ^bb1(%[[VAL_1]] : i64)
// CHECK:         }
func.func @cycle(%arg0: i64, %arg1: i1, %arg2: i64) {
  %alloca = fir.alloca i64
  fir.store %arg2 to %alloca : !fir.ref<i64>
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %use = fir.load %alloca : !fir.ref<i64>
  "test.use"(%use) : (i64) -> ()
  fir.store %arg0 to %alloca : !fir.ref<i64>
  cf.br ^bb2
^bb2:
  cf.br ^bb1
}
