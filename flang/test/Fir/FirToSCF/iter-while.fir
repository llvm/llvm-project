// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @test_simple_iterate_while_1() -> (index, i1, i16, i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 11 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 22 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 123 : i16
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 456 : i32
// CHECK:           %[[WHILE_0:.*]]:4 = scf.while (%[[VAL_0:.*]] = %[[CONSTANT_0]], %[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_4]], %[[VAL_3:.*]] = %[[CONSTANT_5]]) : (index, i1, i16, i32) -> (index, i1, i16, i32) {
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_6]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_2]], %[[CONSTANT_6]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : index, i1, i16, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: i1, %[[VAL_6:.*]]: i16, %[[VAL_7:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_2]] : index
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant true
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant 22 : i16
// CHECK:             %[[CONSTANT_9:.*]] = arith.constant 33 : i32
// CHECK:             scf.yield %[[ADDI_0]], %[[CONSTANT_7]], %[[CONSTANT_8]], %[[CONSTANT_9]] : index, i1, i16, i32
// CHECK:           } attributes {finalValue}
// CHECK:           return %[[VAL_8:.*]]#0, %[[VAL_8]]#1, %[[VAL_8]]#2, %[[VAL_8]]#3 : index, i1, i16, i32
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
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[WHILE_0:.*]]:3 = scf.while (%[[VAL_0:.*]] = %[[ARG0]], %[[VAL_1:.*]] = %[[ARG2]], %[[VAL_2:.*]] = %[[ARG3]]) : (index, i1, i32) -> (index, i1, i32) {
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_1]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : index, i1, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_3:.*]]: index, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 123 : i32
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant true
// CHECK:             scf.yield %[[ADDI_0]], %[[CONSTANT_3]], %[[CONSTANT_2]] : index, i1, i32
// CHECK:           } attributes {finalValue}
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1, %[[VAL_6]]#2 : index, i1, i32
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

// CHECK-LABEL:   func.func @loop_with_negtive_step(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> i1 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[WHILE_0:.*]]:2 = scf.while (%[[VAL_0:.*]] = %[[ARG0]], %[[VAL_1:.*]] = %[[CONSTANT_1]]) : (index, i1) -> (index, i1) {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]] : index, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[VAL_4:.*]] = "test.get_some_value"() : () -> i1
// CHECK:             scf.yield %[[ADDI_0]], %[[VAL_4]] : index, i1
// CHECK:           } attributes {finalValue}
// CHECK:           return %[[VAL_5:.*]]#1 : i1
// CHECK:         }
func.func @loop_with_negtive_step(%lo : index, %up : index) -> i1 {
  %c-1 = arith.constant -1 : index
  %ok1 = arith.constant true
  %res:2 = fir.iterate_while (%i = %lo to %up step %c-1) and (%j = %ok1) -> (index, i1) {
    %ok = "test.get_some_value"() : () -> i1
    fir.result %i, %ok : index, i1
  }
  return %res#1 : i1
}

// CHECK-LABEL:   func.func @loop_with_zero_step(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> i1 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[WHILE_0:.*]]:2 = scf.while (%[[VAL_0:.*]] = %[[ARG0]], %[[VAL_1:.*]] = %[[CONSTANT_1]]) : (index, i1) -> (index, i1) {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]] : index, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[VAL_4:.*]] = "test.get_some_value"() : () -> i1
// CHECK:             scf.yield %[[ADDI_0]], %[[VAL_4]] : index, i1
// CHECK:           } attributes {finalValue}
// CHECK:           return %[[VAL_5:.*]]#1 : i1
// CHECK:         }
func.func @loop_with_zero_step(%lo : index, %up : index) -> i1 {
  %c0 = arith.constant 0 : index
  %ok1 = arith.constant true
  %res:2 = fir.iterate_while (%i = %lo to %up step %c0) and (%j = %ok1) -> (index, i1) {
    %ok = "test.get_some_value"() : () -> i1
    fir.result %i, %ok : index, i1
  }
  return %res#1 : i1
}

// CHECK-LABEL:   func.func @test_zero_iterations() -> (index, i1, i8) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 10 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 42 : i8
// CHECK:           %[[WHILE_0:.*]]:3 = scf.while (%[[VAL_0:.*]] = %[[CONSTANT_0]], %[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[CONSTANT_4]]) : (index, i1, i8) -> (index, i1, i8) {
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_5]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_2]], %[[CONSTANT_5]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : index, i1, i8
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_3:.*]]: index, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i8):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_2]] : index
// CHECK:             scf.yield %[[ADDI_0]], %[[VAL_4]], %[[VAL_5]] : index, i1, i8
// CHECK:           } attributes {finalValue}
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1, %[[VAL_6]]#2 : index, i1, i8
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

// CHECK-LABEL:   func.func @test_without_final_value(
// CHECK-SAME:      %[[ARG0:.*]]: index,
// CHECK-SAME:      %[[ARG1:.*]]: index) -> i1 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[WHILE_0:.*]]:2 = scf.while (%[[VAL_0:.*]] = %[[ARG0]], %[[VAL_1:.*]] = %[[CONSTANT_1]]) : (index, i1) -> (index, i1) {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sle, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:             %[[ORI_0:.*]] = arith.ori %[[ANDI_0]], %[[ANDI_1]] : i1
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[VAL_1]], %[[ORI_0]] : i1
// CHECK:             scf.condition(%[[ANDI_2]]) %[[VAL_0]], %[[VAL_1]] : index, i1
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[VAL_4:.*]] = "test.get_some_value"() : () -> i1
// CHECK:             scf.yield %[[ADDI_0]], %[[VAL_4]] : index, i1
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]]#1 : i1
// CHECK:         }
func.func @test_without_final_value(%lo : index, %up : index) -> i1 {
  %c1 = arith.constant 1 : index
  %ok1 = arith.constant true
  %ok2 = fir.iterate_while (%i = %lo to %up step %c1) and (%j = %ok1) {
    %ok = "test.get_some_value"() : () -> i1
    fir.result %ok : i1
  }
  return %ok2 : i1
}
