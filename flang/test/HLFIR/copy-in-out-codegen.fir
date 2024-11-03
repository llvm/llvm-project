// Test hlfir.copy_in and hlfir.copy_out operation codegen

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_copy_in(%box: !fir.box<!fir.array<?xf64>>) {
  %0:2 = hlfir.copy_in %box : (!fir.box<!fir.array<?xf64>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>) {
// CHECK:    %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
// CHECK:    %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:    %[[VAL_3:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_2]]) : (!fir.box<none>) -> i1
// CHECK:    %[[VAL_4:.*]] = fir.if %[[VAL_3]] -> (!fir.box<!fir.array<?xf64>>) {
// CHECK:      fir.result %[[VAL_0]] : !fir.box<!fir.array<?xf64>>
// CHECK:    } else {
// CHECK:      %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf64>>
// CHECK:      %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:      %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:      %[[VAL_8:.*]] = fir.embox %[[VAL_5]](%[[VAL_7]]) : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf64>>>
// CHECK:      fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:      %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:      %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:      %[[VAL_15:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_12]], %[[VAL_13]],
// CHECK:      %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:      %[[VAL_17:.*]] = fir.rebox %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:      fir.result %[[VAL_17]] : !fir.box<!fir.array<?xf64>>
// CHECK:    }
// CHECK:    %[[VAL_18:.*]] = arith.constant false
// CHECK:    %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_18]] : i1
// CHECK:    return
// CHECK:  }

func.func @test_copy_in_optional(%box: !fir.box<!fir.array<?xf64>>, %is_present: i1) {
  %0:2 = hlfir.copy_in %box handle_optional %is_present : (!fir.box<!fir.array<?xf64>>, i1) -> (!fir.box<!fir.array<?xf64>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in_optional(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK:    %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
// CHECK:    %[[VAL_3:.*]]:2 = fir.if %[[VAL_1]] -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:      %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:      %[[VAL_5:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_4]]) : (!fir.box<none>) -> i1
// CHECK:      %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.box<!fir.array<?xf64>>) {
// CHECK:        fir.result %[[VAL_0]] : !fir.box<!fir.array<?xf64>>
// CHECK:      } else {
// CHECK:        %[[VAL_7:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf64>>
// CHECK:        %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:        %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
// CHECK:        %[[VAL_10:.*]] = fir.embox %[[VAL_7]](%[[VAL_9]]) : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf64>>>
// CHECK:        fir.store %[[VAL_10]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:        %[[VAL_14:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:        %[[VAL_15:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:        %[[VAL_17:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_14]], %[[VAL_15]],
// CHECK:        %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
// CHECK:        %[[VAL_19:.*]] = fir.rebox %[[VAL_18]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:        fir.result %[[VAL_19]] : !fir.box<!fir.array<?xf64>>
// CHECK:      }
// CHECK:      %[[VAL_20:.*]] = arith.constant false
// CHECK:      %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_20]] : i1
// CHECK:      fir.result %[[VAL_22:.*]], %[[VAL_21]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    } else {
// CHECK:      %[[VAL_23:.*]] = fir.absent !fir.box<!fir.array<?xf64>>
// CHECK:      fir.result %[[VAL_23]], %[[VAL_1]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    }

func.func @test_copy_out_no_copy_back(%temp: !fir.box<!fir.array<?xf64>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied : (!fir.box<!fir.array<?xf64>>, i1) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_no_copy_back(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK-NEXT:    fir.if %[[VAL_1]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK-NEXT:      %[[VAL_3:.*]] = fir.call @_FortranADestroyWithoutFinalization(%[[VAL_2]]) : (!fir.box<none>) -> none
// CHECK-NEXT:      %[[VAL_4:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK-NEXT:      fir.freemem %[[VAL_4]] : !fir.heap<!fir.array<?xf64>>
// CHECK-NEXT:    }

func.func @test_copy_out_copy_back(%box: !fir.box<!fir.array<?xf64>>, %temp: !fir.box<!fir.array<?xf64>>, %was_copied: i1) {
  hlfir.copy_out %temp, %was_copied to %box : (!fir.box<!fir.array<?xf64>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_copy_out_copy_back(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_2:.*]]: i1) {
// CHECK:    %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
// CHECK:    fir.if %[[VAL_2]] {
// CHECK:      fir.store %[[VAL_0]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:      %[[VAL_7:.*]] = arith.constant true
// CHECK:      %[[VAL_8:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.array<?xf64>>>) -> !fir.ref<!fir.box<none>>
// CHECK:      %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:      %[[VAL_11:.*]] = fir.call @_FortranACopyOutAssign(%[[VAL_8]], %[[VAL_9]], %[[VAL_7]],
// CHECK:      %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<none>
// CHECK:      %[[VAL_13:.*]] = fir.call @_FortranADestroyWithoutFinalization(%[[VAL_12]]) : (!fir.box<none>) -> none
// CHECK:      %[[VAL_14:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK:      fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<?xf64>>
// CHECK:    }

func.func @test_copy_in_poly(%poly : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) {
  %0:2 = hlfir.copy_in %poly : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, i1)
  return
}
// CHECK-LABEL:   func.func @test_copy_in_poly(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) -> !fir.box<none>
// CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_2]]) : (!fir.box<none>) -> i1
// CHECK:           %[[VAL_4:.*]] = fir.if %[[VAL_3]] -> (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) {
// CHECK:             fir.result %[[VAL_0]] : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:           } else {
// CHECK:             %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_8:.*]] = fir.embox %[[VAL_5]](%[[VAL_7]]) : (!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>
// CHECK:             fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>
// CHECK:             %[[VAL_9:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
// CHECK:             %[[VAL_10:.*]] = arith.constant {{.*}} : index
// CHECK:             %[[VAL_11:.*]] = arith.constant {{.*}} : i32
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             %[[VAL_13:.*]] = fir.convert %[[VAL_0]] : (!fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>) -> !fir.box<none>
// CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_15:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_11]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>>
// CHECK:             %[[VAL_17:.*]] = fir.rebox %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>>) -> !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:             fir.result %[[VAL_17]] : !fir.class<!fir.array<?x!fir.type<test_copy_in_polyTt1{i:i32}>>>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = arith.constant false
// CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_3]], %[[VAL_18]] : i1
// CHECK:           return
// CHECK:         }
