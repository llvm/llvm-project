// Test hlfir.copy_in and hlfir.copy_out operation codegen

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_copy_in(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
  %0:2 = hlfir.copy_in %box to %temp : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_2]]) : (!fir.box<none>) -> i1
// CHECK:           %[[VAL_4:.*]] = fir.if %[[VAL_3]] -> (!fir.box<!fir.array<?xf64>>) {
// CHECK:             fir.result %[[VAL_0]] : !fir.box<!fir.array<?xf64>>
// CHECK:           } else {
// CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:             %[[VAL_11:.*]] = fir.call @_FortranACopyInAssign(%[[VAL_8]], %[[VAL_9]],
// CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:             %[[VAL_13:.*]] = fir.rebox %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:             fir.result %[[VAL_13]] : !fir.box<!fir.array<?xf64>>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = arith.constant false
// CHECK:           %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_14]] : i1
// CHECK:           return
// CHECK:         }

func.func @test_copy_in_optional(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %is_present: i1) {
  %0:2 = hlfir.copy_in %box to %temp handle_optional %is_present : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in_optional(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: i1) {
// CHECK:           %[[VAL_3:.*]]:2 = fir.if %[[VAL_2]] -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:             %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:             %[[VAL_5:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_4]]) : (!fir.box<none>) -> i1
// CHECK:             %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.box<!fir.array<?xf64>>) {
// CHECK:               fir.result %[[VAL_0]] : !fir.box<!fir.array<?xf64>>
// CHECK:             } else {
// CHECK:               %[[VAL_10:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:               %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:               %[[VAL_13:.*]] = fir.call @_FortranACopyInAssign(%[[VAL_10]], %[[VAL_11]],
// CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:               %[[VAL_15:.*]] = fir.rebox %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:               fir.result %[[VAL_15]] : !fir.box<!fir.array<?xf64>>
// CHECK:             }
// CHECK:             %[[VAL_16:.*]] = arith.constant false
// CHECK:             %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_16]] : i1
// CHECK:             fir.result %[[VAL_6]], %[[VAL_17]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:           } else {
// CHECK:             %[[VAL_18:.*]] = fir.absent !fir.box<!fir.array<?xf64>>
// CHECK:             fir.result %[[VAL_18]], %[[VAL_2]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_copy_out_no_copy_back(%temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_no_copy_back(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: i1) {
// CHECK:           fir.if %[[VAL_1]] {
// CHECK:             %[[VAL_2:.*]] = fir.zero_bits !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:             %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_9:.*]] = fir.call @_FortranACopyOutAssign(%[[VAL_6]], %[[VAL_7]],
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_copy_out_copy_back(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied to %box : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_copy_back(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
// CHECK-SAME:                                       %[[VAL_2:.*]]: i1) {
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
// CHECK:           fir.if %[[VAL_2]] {
// CHECK:             fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:             %[[VAL_7:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_10:.*]] = fir.call @_FortranACopyOutAssign(%[[VAL_7]], %[[VAL_8]],
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_copy_in_poly(%poly : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, %temp: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>) {
  %0:2 = hlfir.copy_in %poly to %temp : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>) -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in_poly(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_2]]) : (!fir.box<none>) -> i1
// CHECK:           %[[VAL_4:.*]] = fir.if %[[VAL_3]] -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:           } else {
// CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) -> !fir.box<none>
// CHECK:             %[[VAL_11:.*]] = fir.call @_FortranACopyInAssign(%[[VAL_8]], %[[VAL_9]],
// CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>
// CHECK:             %[[VAL_13:.*]] = fir.rebox %[[VAL_12]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>) -> !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:             fir.result %[[VAL_13]] : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = arith.constant false
// CHECK:           %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_14]] : i1
// CHECK:           return
// CHECK:         }
