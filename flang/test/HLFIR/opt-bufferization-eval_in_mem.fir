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
// CHECK:         fir.save_result %[[VAL_5]] to %[[VAL_4]]#1(%[[VAL_3]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
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
