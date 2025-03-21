// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck %s

// -----

func.func @test_none_innermost(%arg0: !fir.box<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<none>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_none_innermost(
// CHECK-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_0]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }

// -----

func.func @test_none_whole(%arg0: !fir.box<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<none>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_none_whole(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_0]]) : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_1]] : i1
// CHECK:         }

// -----

func.func @test_array_innermost(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_array_innermost(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_2]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// CHECK:           return %[[VAL_3]] : i1
// CHECK:         }

// -----

func.func @test_array_whole(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_array_whole(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }

// -----

func.func @test_assumed_rank_innermost(%arg0: !fir.box<!fir.array<*:f32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<!fir.array<*:f32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_assumed_rank_innermost(
// CHECK-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_2]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// CHECK:           return %[[VAL_3]] : i1
// CHECK:         }

// -----

func.func @test_assumed_rank_whole(%arg0: !fir.box<!fir.array<*:f32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<!fir.array<*:f32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL:   func.func @test_assumed_rank_whole(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) : (!fir.box<none>) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }

// -----

func.func @test_none(%arg0: !fir.box<none>) -> i16 {
  %0 = fir.box_total_elements %arg0 : (!fir.box<none>) -> i16
  return %0 : i16
}
// CHECK-LABEL:   func.func @test_none(
// CHECK-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i16 {
// CHECK:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_5:.*]] = fir.call @_FortranASize(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> i16
// CHECK:           return %[[VAL_6]] : i16
// CHECK:         }

// -----

func.func @test_array(%arg0: !fir.box<!fir.array<?x?xf32>>) -> i32 {
  %0 = fir.box_total_elements %arg0 : (!fir.box<!fir.array<?x?xf32>>) -> i32
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_array(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>>) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_1:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_6:.*]] = fir.call @_FortranASize(%[[VAL_4]], %[[VAL_5]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> i32
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }

// -----

func.func @test_assumed_rank(%arg0: !fir.box<!fir.array<*:f32>>) -> index {
  %0 = fir.box_total_elements %arg0 : (!fir.box<!fir.array<*:f32>>) -> index
  return %0 : index
}
// CHECK-LABEL:   func.func @test_assumed_rank(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> index {
// CHECK:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// CHECK:           %[[VAL_1:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_6:.*]] = fir.call @_FortranASize(%[[VAL_4]], %[[VAL_5]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// CHECK:           return %[[VAL_7]] : index
// CHECK:         }
