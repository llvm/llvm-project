// RUN: fir-opt --canonicalize %s | FileCheck %s

// No folding for !fir.box<none>.
func.func @test_none(%arg0: !fir.box<none>) -> index {
  %0 = fir.box_total_elements %arg0 : (!fir.box<none>) -> index
  return %0 : index
}
// CHECK-LABEL:   func.func @test_none(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> index {
// CHECK:           %[[VAL_1:.*]] = fir.box_total_elements %[[VAL_0]] : (!fir.box<none>) -> index
// CHECK:           return %[[VAL_1]] : index
// CHECK:         }

// No folding for assumed rank arrays.
func.func @test_assumed_rank(%arg0: !fir.box<!fir.array<*:f32>>) -> index {
  %0 = fir.box_total_elements %arg0 : (!fir.box<!fir.array<*:f32>>) -> index
  return %0 : index
}
// CHECK-LABEL:   func.func @test_assumed_rank(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> index {
// CHECK:           %[[VAL_1:.*]] = fir.box_total_elements %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> index
// CHECK:           return %[[VAL_1]] : index
// CHECK:         }

// Scalar boxes hold one element.
func.func @test_scalar_i32(%arg0: !fir.class<!fir.type<sometype{i:i32}>>) -> i32 {
  %0 = fir.box_total_elements %arg0 : (!fir.class<!fir.type<sometype{i:i32}>>) -> i32
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_scalar_i32(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.type<sometype{i:i32}>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }

// Scalar boxes hold one element.
func.func @test_scalar_index(%arg0: !fir.class<!fir.type<sometype{i:i32}>>) -> index {
  %0 = fir.box_total_elements %arg0 : (!fir.class<!fir.type<sometype{i:i32}>>) -> index
  return %0 : index
}
// CHECK-LABEL:   func.func @test_scalar_index(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.type<sometype{i:i32}>>) -> index {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           return %[[VAL_1]] : index
// CHECK:         }
