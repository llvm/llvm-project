// RUN: fir-opt %s --fir-to-scf | FileCheck %s

// CHECK-LABEL:   func.func @test_simple_iterate_while_1() -> (index, i1, i16, i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 11 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 22 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 123 : i16
// CHECK:           %[[VAL_5:.*]] = arith.constant 456 : i32
// CHECK:           %[[VAL_6:.*]]:4 = scf.while (%[[VAL_7:.*]] = %[[VAL_0]], %[[VAL_8:.*]] = %[[VAL_3]], %[[VAL_9:.*]] = %[[VAL_4]], %[[VAL_10:.*]] = %[[VAL_5]]) : (index, i1, i16, i32) -> (index, i1, i16, i32) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi sle, %[[VAL_7]], %[[VAL_1]] : index
// CHECK:             %[[VAL_12:.*]] = arith.andi %[[VAL_11]], %[[VAL_8]] : i1
// CHECK:             scf.condition(%[[VAL_12]]) %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : index, i1, i16, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: i1, %[[VAL_15:.*]]: i16, %[[VAL_16:.*]]: i32):
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_2]] : index
// CHECK:             %[[VAL_18:.*]] = arith.constant true
// CHECK:             %[[VAL_19:.*]] = arith.constant 22 : i16
// CHECK:             %[[VAL_20:.*]] = arith.constant 33 : i32
// CHECK:             scf.yield %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]] : index, i1, i16, i32
// CHECK:           }
// CHECK:           return %[[VAL_21:.*]]#0, %[[VAL_21]]#1, %[[VAL_21]]#2, %[[VAL_21]]#3 : index, i1, i16, i32
// CHECK:         }
func.func @test_simple_iterate_while_1() -> (index, i1, i16, i32) {
  %lo = arith.constant 11 : index
  %up = arith.constant 22 : index
  %step = arith.constant 2 : index
  %ok = arith.constant 1 : i1
  %val1 = arith.constant 123 : i16
  %val2 = arith.constant 456 : i32

  %res:4 = fir.iterate_while (%i = %lo to %up step %step) and (%c = %ok) iter_args(%v1 = %val1, %v2 = %val2) -> (index, i1, i16, i32) {
    %new_c = arith.constant 1 : i1
    %new_v1 = arith.constant 22 : i16
    %new_v2 = arith.constant 33 : i32
    fir.result %i, %new_c, %new_v1, %new_v2 : index, i1, i16, i32
  }

  return %res#0, %res#1, %res#2, %res#3 : index, i1, i16, i32
}

// CHECK-LABEL:   func.func @test_simple_iterate_while_2(
// CHECK-SAME:        %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: i32) -> (index, i1, i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]]:3 = scf.while (%[[VAL_2:.*]] = %[[ARG0]], %[[VAL_3:.*]] = %[[ARG2]], %[[VAL_4:.*]] = %[[ARG3]]) : (index, i1, i32) -> (index, i1, i32) {
// CHECK:             %[[VAL_5:.*]] = arith.cmpi sle, %[[VAL_2]], %[[ARG1]] : index
// CHECK:             %[[VAL_6:.*]] = arith.andi %[[VAL_5]], %[[VAL_3]] : i1
// CHECK:             scf.condition(%[[VAL_6]]) %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : index, i1, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index, %[[VAL_8:.*]]: i1, %[[VAL_9:.*]]: i32):
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_0]] : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 123 : i32
// CHECK:             %[[VAL_12:.*]] = arith.constant true
// CHECK:             scf.yield %[[VAL_10]], %[[VAL_12]], %[[VAL_11]] : index, i1, i32
// CHECK:           }
// CHECK:           return %[[VAL_13:.*]]#0, %[[VAL_13]]#1, %[[VAL_13]]#2 : index, i1, i32
// CHECK:         }
func.func @test_simple_iterate_while_2(%start: index, %stop: index, %cond: i1, %val: i32) -> (index, i1, i32) {
  %step = arith.constant 1 : index

  %res:3 = fir.iterate_while (%i = %start to %stop step %step) and (%ok = %cond) iter_args(%x = %val) -> (index, i1, i32) {
    %new_x = arith.constant 123 : i32
    %new_ok = arith.constant 1 : i1
    fir.result %i, %new_ok, %new_x : index, i1, i32
  }

  return %res#0, %res#1, %res#2 : index, i1, i32
}

// CHECK-LABEL:   func.func @test_zero_iterations() -> (index, i1, i8) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 42 : i8
// CHECK:           %[[VAL_5:.*]]:3 = scf.while (%[[VAL_6:.*]] = %[[VAL_0]], %[[VAL_7:.*]] = %[[VAL_3]], %[[VAL_8:.*]] = %[[VAL_4]]) : (index, i1, i8) -> (index, i1, i8) {
// CHECK:             %[[VAL_9:.*]] = arith.cmpi sle, %[[VAL_6]], %[[VAL_1]] : index
// CHECK:             %[[VAL_10:.*]] = arith.andi %[[VAL_9]], %[[VAL_7]] : i1
// CHECK:             scf.condition(%[[VAL_10]]) %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : index, i1, i8
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: i1, %[[VAL_13:.*]]: i8):
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : index
// CHECK:             scf.yield %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index, i1, i8
// CHECK:           }
// CHECK:           return %[[VAL_15:.*]]#0, %[[VAL_15]]#1, %[[VAL_15]]#2 : index, i1, i8
// CHECK:         }
func.func @test_zero_iterations() -> (index, i1, i8) {
  %lo = arith.constant 10 : index
  %up = arith.constant 5 : index
  %step = arith.constant 1 : index
  %ok = arith.constant 1 : i1
  %x = arith.constant 42 : i8

  %res:3 = fir.iterate_while (%i = %lo to %up step %step) and (%c = %ok) iter_args(%xv = %x) -> (index, i1, i8) {
    fir.result %i, %c, %xv : index, i1, i8
  }

  return %res#0, %res#1, %res#2 : index, i1, i8
}
