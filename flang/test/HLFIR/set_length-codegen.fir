// Test hlfir.set_length operation lowering to operations operating on memory.
// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @test_cst_len(%str : !fir.boxchar<1>) {
  %c10 = arith.constant 10 : index
  %0 = hlfir.set_length %str len %c10 : (!fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @test_cst_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,10> {bindc_name = ".tmp"}
// CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_2]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:  hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : !fir.boxchar<1>, !fir.ref<!fir.char<1,10>>
// CHECK:  %[[VAL_4:.*]] = arith.constant false
// CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_4]], [1 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, !fir.ref<!fir.char<1,10>>) -> tuple<!fir.ref<!fir.char<1,10>>, i1>

func.func @test_dyn_len(%str : !fir.ref<!fir.char<1,10>>, %len : index) {
  %0 = hlfir.set_length %str len %len : (!fir.ref<!fir.char<1,10>>, index) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @test_dyn_len(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_1]] : index) {bindc_name = ".tmp"}
// CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_1]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : !fir.ref<!fir.char<1,10>>, !fir.boxchar<1>
// CHECK:  %[[VAL_4:.*]] = arith.constant false
// CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_4]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
