! Test lowering of transformational intrinsic to HLFIR what matters here
! is not to test each transformational, but to check how their
! lowering interfaces with the rest of lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_transformational_implemented_with_runtime_allocation(x)
  real :: x(10, 10)
  ! MINLOC result is allocated inside the runtime and returned in
  ! a descriptor that was passed by reference to the runtime.
  ! Lowering does the following:
  !  - declares the temp created by the runtime as an hlfir variable.
  !  - "moves" this variable to an hlfir.expr
  !  - associate the expression to takes_array_arg dummy argument
  !  - destroys the expression after the call.

  ! After bufferization, this will allow the buffer created by the
  ! runtime to be passed to takes_array_arg without creating any
  ! other temporaries and to be deallocated after the call.
  call takes_array_arg(minloc(x))
end subroutine
! CHECK-LABEL: func.func @_QPtest_transformational_implemented_with_runtime_allocation(
! CHECK-SAME:                                                                          %[[VAL_0:.*]]: !fir.ref<!fir.array<10x10xf32>> {fir.bindc_name = "x"}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_22:.*]] = fir.call @_FortranAMinlocReal4(%[[VAL_17]], {{.*}}
! CHECK:  %[[VAL_23:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_26:.*]] = fir.box_addr %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:  %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_26]](%{{.*}}) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:  %[[VAL_29:.*]] = arith.constant true
! CHECK:  %[[VAL_30:.*]] = hlfir.as_expr %[[VAL_28]]#0 move %[[VAL_29]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:  %[[VAL_32:.*]]:3 = hlfir.associate %[[VAL_30]](%{{.*}}) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:  %[[VAL_33:.*]] = fir.convert %[[VAL_32]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:  fir.call @_QPtakes_array_arg(%[[VAL_33]])
! CHECK:  hlfir.end_associate %[[VAL_32]]#1, %[[VAL_32]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:  hlfir.destroy %[[VAL_30]] : !hlfir.expr<?xi32>
