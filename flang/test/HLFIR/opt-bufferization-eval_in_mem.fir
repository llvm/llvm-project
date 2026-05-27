// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Fortran F2023 15.5.2.14 point 4. ensures that _QPfoo cannot access _QFtestEx
// and the temporary storage for the result can be avoided.
func.func @_QPtest(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  return
}
func.func private @_QPfoo() -> !fir.array<10xf32>

// CHECK-LABEL: func.func @_QPtest(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x"}) {
// CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:         %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:         %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
// CHECK:         %[[VAL_5:.*]] = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
// CHECK:         fir.save_result %[[VAL_5]] to %[[VAL_4]]#0(%[[VAL_3]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
// CHECK:         return
// CHECK:       }


// Temporary storage cannot be avoided in this case since
// _QFnegative_test_is_targetEx has the TARGET attribute.
func.func @_QPnegative_test_is_target(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x", fir.target}) {
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFnegative_test_is_targetEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @_QPnegative_test_is_target(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x", fir.target}) {
// CHECK:         %[[VAL_2:.*]] = arith.constant false
// CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.array<10xf32>
// CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]]{{.*}}
// CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]]{{.*}}
// CHECK:         %[[VAL_9:.*]] = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
// CHECK:         fir.save_result %[[VAL_9]] to %[[VAL_8]]#1{{.*}}
// CHECK:         %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_8]]#0 move %[[VAL_2]] : (!fir.ref<!fir.array<10xf32>>, i1) -> !hlfir.expr<10xf32>
// CHECK:         hlfir.assign %[[VAL_10]] to %[[VAL_7]]#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
// CHECK:         hlfir.destroy %[[VAL_10]] : !hlfir.expr<10xf32>
// CHECK:         return
// CHECK:       }


// A local (Allocate) TARGET variable whose address is not bound to a Fortran
// POINTER anywhere in the surrounding procedure can be evaluated in place:
// _QPfoo cannot reach it indirectly, matching classic nvfortran's behaviour
// for `lhs = func()` with a procedure-local TARGET LHS.
func.func @_QPpositive_test_target_local_no_capture() {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32> {bindc_name = "x", fir.target, uniq_name = "_QFpositive_test_target_local_no_captureEx"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFpositive_test_target_local_no_captureEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @_QPpositive_test_target_local_no_capture(
// CHECK:         %[[VAL_X:.*]] = fir.alloca !fir.array<10xf32>
// CHECK:         %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_X]]{{.*}}
// CHECK:         %[[VAL_CALL:.*]] = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
// CHECK:         fir.save_result %[[VAL_CALL]] to %[[VAL_DECL]]#0{{.*}}
// CHECK-NOT:     hlfir.as_expr
// CHECK:         return


// Mirrors the RRTMGP `col_dry_arr` pattern: a local TARGET LHS bound to a
// POINTER *after* the call. The pointer binding cannot have been observed by
// the call, so the temporary can still be elided.
func.func @_QPpositive_test_target_local_captured_after(
    %arg_ptr: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32> {bindc_name = "x", fir.target, uniq_name = "_QFpositive_test_target_local_captured_afterEx"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFpositive_test_target_local_captured_afterEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  %5 = fir.convert %2#0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %6 = fir.embox %5(%1) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %6 to %arg_ptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  return
}
// CHECK-LABEL: func.func @_QPpositive_test_target_local_captured_after(
// CHECK:         %[[VAL_X:.*]] = fir.alloca !fir.array<10xf32>
// CHECK:         %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_X]]{{.*}}
// CHECK:         %[[VAL_CALL:.*]] = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
// CHECK:         fir.save_result %[[VAL_CALL]] to %[[VAL_DECL]]#0{{.*}}
// CHECK-NOT:     hlfir.as_expr
// CHECK:         fir.embox
// CHECK:         return


// A local TARGET variable whose address is bound to a Fortran POINTER *before*
// the call. The temporary must be preserved because _QPfoo could observe the
// LHS through that pointer.
func.func @_QPnegative_test_target_local_captured_before(
    %arg_ptr: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32> {bindc_name = "x", fir.target, uniq_name = "_QFnegative_test_target_local_captured_beforeEx"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFnegative_test_target_local_captured_beforeEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %5 = fir.convert %2#0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %6 = fir.embox %5(%1) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %6 to %arg_ptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  return
}
// CHECK-LABEL: func.func @_QPnegative_test_target_local_captured_before(
// CHECK:         fir.alloca !fir.array<10xf32>
// CHECK:         %[[TMP:.*]] = fir.alloca !fir.array<10xf32>
// CHECK:         hlfir.as_expr
// CHECK:         return


// A multi-block (post-CFG) function: the capturing fir.embox is in a later
// block than `call`. The "use textually after op in the same block" skip rule
// is unsound here because the cf.br could form a back-edge, so the temporary
// must be preserved.
func.func @_QPnegative_test_target_local_multi_block(
    %arg_ptr: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.alloca !fir.array<10xf32> {bindc_name = "x", fir.target, uniq_name = "_QFnegative_test_target_local_multi_blockEx"}
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %0(%1) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFnegative_test_target_local_multi_blockEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg1: !fir.ref<!fir.array<10xf32>>):
    %4 = fir.call @_QPfoo() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %4 to %arg1(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %3 to %2#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %3 : !hlfir.expr<10xf32>
  cf.br ^bb1
^bb1:
  %5 = fir.convert %2#0 : (!fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %6 = fir.embox %5(%1) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  fir.store %6 to %arg_ptr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  return
}
// CHECK-LABEL: func.func @_QPnegative_test_target_local_multi_block(
// CHECK:         fir.alloca !fir.array<10xf32>
// CHECK:         %[[TMP:.*]] = fir.alloca !fir.array<10xf32>
// CHECK:         hlfir.as_expr
// CHECK:         fir.embox
