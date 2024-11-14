// Test fir.is_assumed_size lowering to runtime calls in fir-assumed-rank-op pass.
// RUN: fir-opt -o - --fir-assumed-rank-op %s | FileCheck %s

func.func @test_ranked(%arg0: !fir.class<!fir.array<?xnone>> ) -> i1 {
  %1 = fir.is_assumed_size %arg0 : (!fir.class<!fir.array<?xnone>>) -> i1
  return %1 : i1
}

func.func @test_assumed_rank(%arg0: !fir.box<!fir.array<*:f32>> ) -> i1 {
  %1 = fir.is_assumed_size %arg0 : (!fir.box<!fir.array<*:f32>>) -> i1
  return %1 : i1
}
// CHECK-LABEL:   func.func @test_ranked(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.class<!fir.array<?xnone>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?xnone>>) -> !fir.box<none>
// CHECK:           %[[VAL_2:.*]] = fir.call @_FortranAIsAssumedSize(%[[VAL_1]]) : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @test_assumed_rank(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           %[[VAL_2:.*]] = fir.call @_FortranAIsAssumedSize(%[[VAL_1]]) : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }
