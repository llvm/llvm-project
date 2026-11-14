// Test canonicalization of character expressions that are only destroyed.
// RUN: fir-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @only_destroyed
// CHECK-NOT: hlfir.set_length
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @only_destroyed(%a: !fir.boxchar<1>, %n: index) {
  %0 = hlfir.set_length %a len %n : (!fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}

// Erasing an expression can make its operand only destroyed.
// CHECK-LABEL: func.func @nested
// CHECK-NOT: hlfir.concat
// CHECK-NOT: hlfir.set_length
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @nested(%a: !fir.boxchar<1>, %b: !fir.boxchar<1>, %n: index, %m: index) {
  %0 = hlfir.concat %a, %b len %n : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %1 = hlfir.set_length %0 len %m : (!hlfir.expr<!fir.char<1,?>>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %1 : !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL: func.func @char_trim_and_extremum
// CHECK-NOT: hlfir.char_trim
// CHECK-NOT: hlfir.char_extremum
// CHECK-NOT: hlfir.destroy
// CHECK: return
func.func @char_trim_and_extremum(%a: !fir.boxchar<1>, %b: !fir.boxchar<1>) {
  %0 = hlfir.char_trim %a : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  %1 = hlfir.char_extremum max, %a, %b : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %1 : !hlfir.expr<!fir.char<1,?>>
  return
}

// Other expression producers may reuse storage that the hlfir.destroy
// releases, like the storage moved into this hlfir.as_expr.
// CHECK-LABEL: func.func @as_expr_move
// CHECK: %[[EXPR:.*]] = hlfir.as_expr %{{.*}} move
// CHECK: hlfir.destroy %[[EXPR]]
func.func @as_expr_move(%a: !fir.heap<!fir.char<1,?>>, %n: index) {
  %0:2 = hlfir.declare %a typeparams %n uniq_name("tmp") : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
  %true = arith.constant true
  %1 = hlfir.as_expr %0#0 move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.destroy %1 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK-LABEL: func.func @used
// CHECK: %[[EXPR:.*]] = hlfir.concat
// CHECK: hlfir.assign %[[EXPR]]
// CHECK: hlfir.destroy %[[EXPR]]
func.func @used(%a: !fir.boxchar<1>, %b: !fir.boxchar<1>, %out: !fir.boxchar<1>, %n: index) {
  %0 = hlfir.concat %a, %b len %n : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %0 to %out : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}
