// Test hlfir.as_expr code generation

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @char_expr(%addr: !fir.ref<!fir.char<1,?>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %1 = hlfir.as_expr %0#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @char_expr(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,?>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_1]] : index) {bindc_name = ".tmp"}
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_1]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_5:.*]] = arith.constant false
// CHECK:  hlfir.assign %[[VAL_2]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:  %[[VAL_6:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_5]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>

func.func @char_expr_2(%addr: !fir.ref<!fir.char<1,10>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %1 = hlfir.as_expr %0#0 : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @char_expr_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.char<1,10>>,
// CHECK-SAME:    %[[VAL_1:.*]]: index) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.char<1,10> {bindc_name = ".tmp"}
// CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_1]] {uniq_name = "c"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_1]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:  %[[VAL_5:.*]] = arith.constant false
// CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_4]]#0 : !fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>
// CHECK:  %[[VAL_6:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_5]], [1 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,10>>, i1>
// CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,10>>, i1>, !fir.ref<!fir.char<1,10>>) -> tuple<!fir.ref<!fir.char<1,10>>, i1>
