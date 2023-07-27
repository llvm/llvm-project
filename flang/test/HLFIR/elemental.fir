// Test hlfir.elemental and hlfir.yield_element operation parse, verify
// (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @numeric_type(%x: !fir.ref<!fir.array<10x20xi32>>, %y: !fir.ref<!fir.array<10x20xi32>>) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %3 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
  ^bb0(%i: index, %j: index):
    %4 = hlfir.designate %x (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %5 = hlfir.designate %y (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
    %6 = fir.load %4 : !fir.ref<i32>
    %7 = fir.load %5 : !fir.ref<i32>
    %8 = arith.addi %6, %7 : i32
    hlfir.yield_element %8 : i32
  }
  return
}
// CHECK-LABEL:   func.func @numeric_type(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.ref<!fir.array<10x20xi32>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: !fir.ref<!fir.array<10x20xi32>>
// CHECK:  %[[VAL_4:.*]] = fir.shape
// CHECK:  %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
// CHECK:  ^bb0(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// CHECK:    %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_6]], %[[VAL_7]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_6]], %[[VAL_7]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:    %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:    hlfir.yield_element %[[VAL_12]] : i32
// CHECK:  }

func.func @char_type(%x: !fir.box<!fir.array<?x!fir.char<1,?>>>, %n: index, %l : index, %l0 :index) {
  %0 = fir.shape %n : (index) -> !fir.shape<1>
  %3 = hlfir.elemental %0 typeparams %l : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
  ^bb0(%i: index):
    %4 = hlfir.designate %x (%i) typeparams %l0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
    %6 = hlfir.concat %4, %4 len %l : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
    hlfir.yield_element %6 : !hlfir.expr<!fir.char<1,?>>
  }
  return
}
// CHECK-LABEL:   func.func @char_type(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_2:[^:]*]]: index,
// CHECK-SAME:    %[[VAL_3:[^:]*]]: index) {
// CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] typeparams %[[VAL_2]] : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
// CHECK:  ^bb0(%[[VAL_6:.*]]: index):
// CHECK:    %[[VAL_7:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_6]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:    %[[VAL_8:.*]] = hlfir.concat %[[VAL_7]], %[[VAL_7]] len %[[VAL_2]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:    hlfir.yield_element %[[VAL_8]] : !hlfir.expr<!fir.char<1,?>>
// CHECK:  }

!pdt = !fir.type<pdt(param:i32){field:f32}>
func.func @parametrized_derived_transpose(%x: !fir.box<!fir.array<?x?x!pdt>>, %n: index, %m: index, %l: i32) {
  %0 = fir.shape %m, %n : (index, index) -> !fir.shape<2>
  %3 = hlfir.elemental %0 typeparams %l : (!fir.shape<2>, i32) -> !hlfir.expr<?x?x!pdt> {
  ^bb0(%j: index, %i: index):
    %4 = hlfir.designate %x (%j, %i) typeparams %l : (!fir.box<!fir.array<?x?x!pdt>>, index, index, i32) -> !fir.box<!pdt>
    %5 = hlfir.as_expr %4 : (!fir.box<!pdt>) -> !hlfir.expr<!pdt>
    hlfir.yield_element %5 : !hlfir.expr<!pdt>
  }
  return
}
// CHECK-LABEL:   func.func @parametrized_derived_transpose(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.box
// CHECK-SAME: %[[VAL_3:[^:]*]]: i32
// CHECK:  %[[VAL_4:.*]] = fir.shape
// CHECK:  %[[VAL_5:.*]] = hlfir.elemental %[[VAL_4]] typeparams %[[VAL_3]] : (!fir.shape<2>, i32) -> !hlfir.expr<?x?x!fir.type<pdt(param:i32){field:f32}>> {
// CHECK:  ^bb0(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// CHECK:    %[[VAL_8:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_6]], %[[VAL_7]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.type<pdt(param:i32){field:f32}>>>, index, index, i32) -> !fir.box<!fir.type<pdt(param:i32){field:f32}>>
// CHECK:    %[[VAL_9:.*]] = hlfir.as_expr %[[VAL_8]] : (!fir.box<!fir.type<pdt(param:i32){field:f32}>>) -> !hlfir.expr<!fir.type<pdt(param:i32){field:f32}>>
// CHECK:    hlfir.yield_element %[[VAL_9]] : !hlfir.expr<!fir.type<pdt(param:i32){field:f32}>>
// CHECK:  }

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
// CHECK:           %[[VAL_3:.*]] = hlfir.elemental %[[VAL_2]] unordered : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
// CHECK:           ^bb0(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:             hlfir.yield_element %[[VAL_6]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
