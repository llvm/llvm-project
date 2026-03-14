// Test hlfir.destroy code generation and hlfir.yield_element "implicit
// hlfir.destroy" aspect.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @test_move_with_cleanup(%arg0 : !fir.ref<!fir.array<100xi32>>) {
  %must_free = arith.constant true
  %expr = hlfir.as_expr %arg0 move %must_free: (!fir.ref<!fir.array<100xi32>>, i1) -> !hlfir.expr<100xi32>
  hlfir.destroy %expr : !hlfir.expr<100xi32>
  return
}
// CHECK-LABEL:   func.func @test_move_with_cleanup(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>) {
// CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<100xi32>>) -> !fir.heap<!fir.array<100xi32>>
// CHECK:  fir.freemem %[[VAL_5]] : !fir.heap<!fir.array<100xi32>>

func.func @test_move_no_cleanup(%arg0 : !fir.ref<!fir.array<100xi32>>) {
  %must_free = arith.constant false
  %expr = hlfir.as_expr %arg0 move %must_free: (!fir.ref<!fir.array<100xi32>>, i1) -> !hlfir.expr<100xi32>
  hlfir.destroy %expr : !hlfir.expr<100xi32>
  return
}
// CHECK-LABEL:   func.func @test_move_no_cleanup(
// CHECK-NOT:  fir.freemem
// CHECK:  return

func.func @test_elemental() {
  %c100 = arith.constant 100 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3 = hlfir.elemental %0 typeparams %c20 : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,20>> {
  ^bb0(%i: index):
    %buffer = fir.allocmem !fir.char<1,20>
    %must_free = arith.constant true
    %expr = hlfir.as_expr %buffer move %must_free: (!fir.heap<!fir.char<1,20>>, i1) -> !hlfir.expr<!fir.char<1,20>>
    hlfir.yield_element %expr : !hlfir.expr<!fir.char<1,20>>
  }
  return
}
// CHECK-LABEL:   func.func @test_elemental(
// CHECK:  fir.do_loop
// CHECK:    %[[VAL_9:.*]] = fir.allocmem !fir.char<1,20>
// CHECK:    hlfir.assign %[[VAL_9]] to %{{.*}} : !fir.heap<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>
// CHECK:    fir.freemem %[[VAL_9]] : !fir.heap<!fir.char<1,20>>
// CHECK:  }
// CHECK:  return

func.func @test_elemental_expr_created_outside_of_loops() {
  %buffer = fir.allocmem !fir.char<1,20>
  %must_free = arith.constant true
  %expr = hlfir.as_expr %buffer move %must_free: (!fir.heap<!fir.char<1,20>>, i1) -> !hlfir.expr<!fir.char<1,20>>
  %c100 = arith.constant 100 : index
  %c20 = arith.constant 20 : index
  %0 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3 = hlfir.elemental %0 typeparams %c20 : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,20>> {
  ^bb0(%i: index):
    // No freemem should be inserted inside the loops.
    hlfir.yield_element %expr : !hlfir.expr<!fir.char<1,20>>
  }
  hlfir.destroy %expr : !hlfir.expr<!fir.char<1,20>>
  return
}
// CHECK-LABEL:   func.func @test_elemental_expr_created_outside_of_loops() {
// CHECK:  %[[VAL_9:.*]] = fir.allocmem !fir.char<1,20>
// CHECK:  fir.do_loop
// CHECK:    hlfir.assign %[[VAL_9]] to %{{.*}} : !fir.heap<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>
// CHECK-NOT:  fir.freemem
// CHECK:  }
// CHECK:  fir.freemem %[[VAL_9]] : !fir.heap<!fir.char<1,20>>
// CHECK:  return
