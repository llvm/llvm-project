// RUN: fir-opt --canonicalize %s | FileCheck %s

// No folding for !fir.box<none>.
func.func @test_none(%arg0: !fir.box<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<none>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_none(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.is_contiguous_box %[[VAL_0]] whole : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_1]] : i1
// CHECK:         }

// No folding for assumed rank arrays.
func.func @test_assumed_rank(%arg0: !fir.box<!fir.array<*:f32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<!fir.array<*:f32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_assumed_rank(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.is_contiguous_box %[[VAL_0]] innermost : (!fir.box<!fir.array<*:f32>>) -> i1
// CHECK:           return %[[VAL_1]] : i1
// CHECK:         }

// Scalars are always contiguous.
func.func @test_scalar(%arg0: !fir.class<!fir.type<sometype{i:i32}>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.class<!fir.type<sometype{i:i32}>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_scalar(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<!fir.type<sometype{i:i32}>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           return %[[VAL_1]] : i1
// CHECK:         }
