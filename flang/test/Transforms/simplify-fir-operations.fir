// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck --check-prefixes=ALL,NOOPT %s
// RUN: fir-opt --split-input-file --simplify-fir-operations=prefer-inline-implementation=true %s | FileCheck --check-prefixes=ALL,OPT %s

func.func @test_none_innermost(%arg0: !fir.box<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<none>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_none_innermost(
// ALL-SAME:                                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i1 {
// ALL:           %[[VAL_1:.*]] = arith.constant 1 : i32
// ALL:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_0]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// ALL:           return %[[VAL_2]] : i1

// -----

func.func @test_none_whole(%arg0: !fir.box<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<none>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_none_whole(
// ALL-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i1 {
// ALL:           %[[VAL_1:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_0]]) : (!fir.box<none>) -> i1
// ALL:           return %[[VAL_1]] : i1
// ALL:         }

// -----

func.func @test_array_innermost(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_array_innermost(
// ALL-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>>) -> i1 {
// NOOPT:           %[[VAL_1:.*]] = arith.constant 1 : i32
// NOOPT:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// NOOPT:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_2]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// NOOPT:           return %[[VAL_3]] : i1
// OPT:           %[[VAL_1:.*]] = arith.constant 0 : index
// OPT:           %[[VAL_2:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> index
// OPT:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// OPT:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_3]]#2 : index
// OPT:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_3]]#1, %[[VAL_1]] : index
// OPT:           %[[VAL_6:.*]] = arith.ori %[[VAL_4]], %[[VAL_5]] : i1
// OPT:           return %[[VAL_6]] : i1

// -----

func.func @test_array_whole(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_array_whole(
// ALL-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>>) -> i1 {
// NOOPT:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// NOOPT:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) : (!fir.box<none>) -> i1
// NOOPT:           return %[[VAL_2]] : i1
// OPT:           %[[VAL_1:.*]] = arith.constant 0 : index
// OPT:           %[[VAL_2:.*]] = fir.box_elesize %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> index
// OPT:           %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// OPT:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_3]]#2 : index
// OPT:           %[[VAL_5:.*]] = arith.cmpi eq, %[[VAL_3]]#1, %[[VAL_1]] : index
// OPT:           %[[VAL_6:.*]] = arith.ori %[[VAL_4]], %[[VAL_5]] : i1
// OPT:           return %[[VAL_6]] : i1

// -----

func.func @test_assumed_rank_innermost(%arg0: !fir.box<!fir.array<*:f32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.box<!fir.array<*:f32>>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_assumed_rank_innermost(
// ALL-SAME:                                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// ALL:           %[[VAL_1:.*]] = arith.constant 1 : i32
// ALL:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// ALL:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_2]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// ALL:           return %[[VAL_3]] : i1

// -----

func.func @test_assumed_rank_whole(%arg0: !fir.box<!fir.array<*:f32>>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 whole : (!fir.box<!fir.array<*:f32>>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_assumed_rank_whole(
// ALL-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> i1 {
// ALL:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// ALL:           %[[VAL_2:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_1]]) : (!fir.box<none>) -> i1
// ALL:           return %[[VAL_2]] : i1
// ALL:         }

// -----

func.func @test_scalar_upoly(%arg0: !fir.class<none>) -> i1 {
  %0 = fir.is_contiguous_box %arg0 innermost : (!fir.class<none>) -> i1
  return %0 : i1
}
// ALL-LABEL:   func.func @test_scalar_upoly(
// ALL-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.class<none>) -> i1 {
// ALL:           %[[VAL_1:.*]] = arith.constant 1 : i32
// ALL:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.class<none>) -> !fir.box<none>
// ALL:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguousUpTo(%[[VAL_2]], %[[VAL_1]]) : (!fir.box<none>, i32) -> i1
// ALL:           return %[[VAL_3]] : i1

// -----

func.func @test_none(%arg0: !fir.box<none>) -> i16 {
  %0 = fir.box_total_elements %arg0 : (!fir.box<none>) -> i16
  return %0 : i16
}
// ALL-LABEL:   func.func @test_none(
// ALL-SAME:                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<none>) -> i16 {
// ALL:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// ALL:           %[[VAL_1:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// ALL:           %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// ALL:           %[[VAL_5:.*]] = fir.call @_FortranASize(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// ALL:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> i16
// ALL:           return %[[VAL_6]] : i16
// ALL:         }

// -----

func.func @test_array(%arg0: !fir.box<!fir.array<?x?xf32>>) -> i32 {
  %0 = fir.box_total_elements %arg0 : (!fir.box<!fir.array<?x?xf32>>) -> i32
  return %0 : i32
}
// ALL-LABEL:   func.func @test_array(
// ALL-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?xf32>>) -> i32 {
// ALL:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// ALL:           %[[VAL_1:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// ALL:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// ALL:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// ALL:           %[[VAL_6:.*]] = fir.call @_FortranASize(%[[VAL_4]], %[[VAL_5]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// ALL:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> i32
// ALL:           return %[[VAL_7]] : i32
// ALL:         }

// -----

func.func @test_assumed_rank(%arg0: !fir.box<!fir.array<*:f32>>) -> index {
  %0 = fir.box_total_elements %arg0 : (!fir.box<!fir.array<*:f32>>) -> index
  return %0 : index
}
// ALL-LABEL:   func.func @test_assumed_rank(
// ALL-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<*:f32>>) -> index {
// ALL:           %[[VAL_3:.*]] = arith.constant {{.*}} : i32
// ALL:           %[[VAL_1:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// ALL:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>) -> !fir.box<none>
// ALL:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// ALL:           %[[VAL_6:.*]] = fir.call @_FortranASize(%[[VAL_4]], %[[VAL_5]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// ALL:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
// ALL:           return %[[VAL_7]] : index
// ALL:         }
