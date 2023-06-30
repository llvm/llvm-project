// Test hlfir.elemental code generation
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @numeric_type(%arg0: !fir.ref<!fir.array<10x20xi32>>, %arg1: !fir.ref<!fir.array<10x20xi32>>) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %1 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
  ^bb0(%arg2: index, %arg3: index):
    %2 = hlfir.designate %arg0 (%arg2, %arg3)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %3 = hlfir.designate %arg1 (%arg2, %arg3)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %4 = fir.load %2 : !fir.ref<i32>
    %5 = fir.load %3 : !fir.ref<i32>
    %6 = arith.addi %4, %5 : i32
    hlfir.yield_element %6 : i32
  }
  return
}
// CHECK-LABEL:   func.func @numeric_type(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.ref<!fir.array<10x20xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<10x20xi32>>) {
// CHECK:    %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:    %[[VAL_3:.*]] = arith.constant 20 : index
// CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_5:.*]] = fir.allocmem !fir.array<10x20xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>)
// CHECK:    %[[VAL_7:.*]] = arith.constant true
// CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_9:.*]] = %[[VAL_8]] to %[[VAL_3]] step %[[VAL_8]] {
// CHECK:      fir.do_loop %[[VAL_10:.*]] = %[[VAL_8]] to %[[VAL_2]] step %[[VAL_8]] {
// CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        %[[VAL_12:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_10]], %[[VAL_9]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        %[[VAL_13:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
// CHECK:        %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:        %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:        %[[VAL_16:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_10]], %[[VAL_9]])  : (!fir.heap<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:        hlfir.assign %[[VAL_15]] to %[[VAL_16]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:      }
// CHECK:    }
// CHECK:    %[[VAL_17:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_7]], [1 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:    %[[VAL_19:.*]] = fir.insert_value %[[VAL_18]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, !fir.heap<!fir.array<10x20xi32>>) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>


func.func @char_type(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %arg1: index, %arg2: index, %arg3: index) {
  %0 = fir.shape %arg1 : (index) -> !fir.shape<1>
  %1 = hlfir.elemental %0 typeparams %arg2 : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
  ^bb0(%arg4: index):
    %2 = hlfir.designate %arg0 (%arg4)  typeparams %arg3 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
    %3 = hlfir.concat %2, %2 len %arg2 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
    hlfir.yield_element %3 : !hlfir.expr<!fir.char<1,?>>
  }
  return
}
// CHECK-LABEL:   func.func @char_type(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_3:[^:]*]]: index) {
// CHECK:    %[[VAL_4:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_5:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_2]] : index), %[[VAL_1]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_4]]) typeparams %[[VAL_2]] {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>)
// CHECK:    %[[VAL_7:.*]] = arith.constant true
// CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_9:.*]] = %[[VAL_8]] to %[[VAL_1]] step %[[VAL_8]] {
// CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_9]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
               // concatenation
// CHECK:      %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_14:.*]] typeparams %[[VAL_13:.*]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:      %[[VAL_31:.*]] = arith.constant false
// CHECK:      %[[VAL_32:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_33:.*]] = fir.insert_value %[[VAL_32]], %[[VAL_31]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_30]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:      %[[VAL_35:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_9]])  typeparams %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:      hlfir.assign %[[VAL_30]]#0 to %[[VAL_35]] temporary_lhs : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:    }
// CHECK:    %[[VAL_36:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>
// CHECK:    %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_7]], [1 : index] : (tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>
// CHECK:    %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>, !fir.box<!fir.array<?x!fir.char<1,?>>>) -> tuple<!fir.box<!fir.array<?x!fir.char<1,?>>>, i1>


func.func @derived_transpose(%arg0: !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, %arg1: index, %arg2: index) {
  %0 = fir.shape %arg2, %arg1 : (index, index) -> !fir.shape<2>
  %1 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.type<t{field:f32}>> {
  ^bb0(%arg4: index, %arg5: index):
    %2 = hlfir.designate %arg0 (%arg4, %arg5) : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.box<!fir.type<t{field:f32}>>
    %3 = hlfir.as_expr %2 : (!fir.box<!fir.type<t{field:f32}>>) -> !hlfir.expr<!fir.type<t{field:f32}>>
    hlfir.yield_element %3 : !hlfir.expr<!fir.type<t{field:f32}>>
  }
  return
}
// CHECK-LABEL:   func.func @derived_transpose(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_2:.*]]: index) {
// CHECK:    %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:    %[[VAL_4:.*]] = fir.allocmem !fir.array<?x?x!fir.type<t{field:f32}>>, %[[VAL_2]], %[[VAL_1]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x?x!fir.type<t{field:f32}>>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, !fir.heap<!fir.array<?x?x!fir.type<t{field:f32}>>>)
// CHECK:    %[[VAL_6:.*]] = arith.constant true
// CHECK:    %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:    fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_1]] step %[[VAL_7]] {
// CHECK:      fir.do_loop %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_2]] step %[[VAL_7]] {
// CHECK:        %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_9]], %[[VAL_8]])  : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.box<!fir.type<t{field:f32}>>
// CHECK:        %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_8]])  : (!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, index, index) -> !fir.ref<!fir.type<t{field:f32}>>
// CHECK:        hlfir.assign %[[VAL_10]] to %[[VAL_11]] temporary_lhs : !fir.box<!fir.type<t{field:f32}>>, !fir.ref<!fir.type<t{field:f32}>>
// CHECK:      }
// CHECK:    }
// CHECK:    %[[VAL_12:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>
// CHECK:    %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_6]], [1 : index] : (tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>
// CHECK:    %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_5]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>, !fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>) -> tuple<!fir.box<!fir.array<?x?x!fir.type<t{field:f32}>>>, i1>

func.func @unordered() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %3 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
  ^bb0(%i: index, %j: index):
    %c0 = arith.constant 0 : i32
    hlfir.yield_element %c0 : i32
  }
  return
}
// CHECK-LABEL:   func.func @unordered() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_3:.*]] = fir.allocmem !fir.array<10x20xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_2]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<10x20xi32>>, !fir.heap<!fir.array<10x20xi32>>)
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_6]] to %[[VAL_1]] step %[[VAL_6]] unordered {
// CHECK:             fir.do_loop %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_0]] step %[[VAL_6]] unordered {
// CHECK:               %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]], %[[VAL_7]])  : (!fir.heap<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_9]] to %[[VAL_10]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_5]], [1 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x20xi32>>, i1>, !fir.heap<!fir.array<10x20xi32>>) -> tuple<!fir.heap<!fir.array<10x20xi32>>, i1>
// CHECK:           return
// CHECK:         }
