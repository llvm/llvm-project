// RUN: fir-opt %s --canonicalize="region-simplify=disabled" --bufferize-hlfir --canonicalize="region-simplify=disabled" | FileCheck %s

// Explicit cleanup of an intermediate expression survives association through
// no_reassoc and releases the buffer after the comparison.
// CHECK-LABEL: func.func @intermediate
// CHECK: fir.allocmem !fir.char<1,?>
// CHECK: hlfir.cmpchar
// CHECK: fir.freemem
// CHECK-NOT: fir.freemem
// CHECK: return
func.func @intermediate(%a: !fir.boxchar<1>, %b: !fir.boxchar<1>, %c: !fir.boxchar<1>, %n: index) -> i1 {
  %x = hlfir.concat %a, %b len %n : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %alias = hlfir.no_reassoc %x : !hlfir.expr<!fir.char<1,?>>
  %assoc:3 = hlfir.associate %alias typeparams %n : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  %test = hlfir.cmpchar eq %assoc#0 %c : (!fir.boxchar<1>, !fir.boxchar<1>) -> i1
  hlfir.end_associate %assoc#1, %assoc#2 : !fir.ref<!fir.char<1,?>>, i1
  hlfir.destroy %x : !hlfir.expr<!fir.char<1,?>>
  return %test : i1
}

// An explicit cleanup in another CFG block must not be replaced or rejected
// when inspecting the expression's other uses.
// CHECK-LABEL: func.func @existing_cleanup
// CHECK: fir.allocmem !fir.char<1,?>
// CHECK: hlfir.assign
// CHECK: fir.freemem
// CHECK-NOT: fir.freemem
// CHECK: return
func.func @existing_cleanup(%a: !fir.boxchar<1>, %b: !fir.boxchar<1>, %out: !fir.boxchar<1>, %n: index) {
  %x = hlfir.concat %a, %b len %n : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  cf.br ^next
^next:
  hlfir.assign %x to %out : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %x : !hlfir.expr<!fir.char<1,?>>
  return
}
