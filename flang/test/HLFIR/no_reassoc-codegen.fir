// Test hlfir.noreassoc code generation to FIR.

// RUN: fir-opt %s -bufferize-hlfir -convert-hlfir-to-fir | FileCheck %s

func.func @no_reassoc_expr(%addr: !fir.ref<!fir.char<1,?>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %1 = hlfir.as_expr %0#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  %2 = hlfir.no_reassoc %1 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK:  %[[VAL_4:.*]] = fir.alloca !fir.char<1,?>
// CHECK:  %[[VAL_5:.*]] = fir.declare %[[VAL_4]] {{.*}}
// CHECK:  %[[VAL_6:.*]] = fir.emboxchar %[[VAL_5]], %{{.*}}
// CHECK:  fir.no_reassoc %[[VAL_6]] : !fir.boxchar<1>

func.func @no_reassoc_var(%addr: !fir.ref<!fir.char<1,?>>, %len: index) {
  %0:2 = hlfir.declare %addr typeparams %len {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %1 = hlfir.no_reassoc %0#0 : !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @no_reassoc_var(
// CHECK:  %[[VAL_3:.*]] = fir.emboxchar
// CHECK:  fir.no_reassoc %[[VAL_3]] : !fir.boxchar<1>

func.func @no_reassoc_value(%arg0 : i32) {
  %0 = hlfir.no_reassoc %arg0 : i32
  return
}
// CHECK-LABEL:   func.func @no_reassoc_value(
// CHECK-SAME:    %[[VAL_0:.*]]: i32) {
// CHECK:  fir.no_reassoc %[[VAL_0]] : i32
