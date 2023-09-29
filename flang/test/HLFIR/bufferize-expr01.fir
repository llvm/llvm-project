// RUN: fir-opt --bufferize-hlfir --split-input-file %s | FileCheck %s

// -----
// Check that hlfir.as_expr with allocatable operand
// is bufferized with a box load:
func.func @test1() attributes {fir.bindc_name = "test"} {
  %c2_i32 = arith.constant 2 : i32
  %c0_i64 = arith.constant 0 : i64
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %false = arith.constant false
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".result"}
  %1 = fir.call @_QFPfoo1() fastmath<contract> : () -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.save_result %1 to %0 : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %3 = hlfir.as_expr %2#0 move %false : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i1) -> !hlfir.expr<?xi32>
  %9 = fir.shape %c3 : (index) -> !fir.shape<1>
  %10 = hlfir.elemental %9 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg0: index):
    %29 = hlfir.apply %3, %arg0 : (!hlfir.expr<?xi32>, index) -> i32
    %32 = arith.cmpi eq, %29, %c2_i32 : i32
    %33 = fir.convert %32 : (i1) -> !fir.logical<4>
    hlfir.yield_element %33 : !fir.logical<4>
  }
  return
}
// CHECK-LABEL:   func.func @test1() attributes {fir.bindc_name = "test"} {
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = ".result"}
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           fir.do_loop %[[VAL_18:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_18]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
