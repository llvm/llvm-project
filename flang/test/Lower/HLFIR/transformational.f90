! Test lowering of transformational intrinsic to HLFIR what matters here
! is not to test each transformational, but to check how their
! lowering interfaces with the rest of lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_transformational_implemented_with_runtime_allocation(x)
  real :: x(10, 10)
  ! MINLOC result is allocated inside the runtime and returned in
  ! a descriptor that was passed by reference to the runtime.
  ! Lowering goes via a hlfir.minloc intrinsic.

  ! After bufferization, this will allow the buffer created by the
  ! runtime to be passed to takes_array_arg without creating any
  ! other temporaries and to be deallocated after the call.
  call takes_array_arg(minloc(x))
end subroutine
! CHECK-LABEL: func.func @_QPtest_transformational_implemented_with_runtime_allocation(
! CHECK-SAME:                                                                          %[[ARG0:.*]]: !fir.ref<!fir.array<10x10xf32>> {fir.bindc_name = "x"}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_transformational_implemented_with_runtime_allocationEx"}
! CHECK:  %[[VAL_2:.*]] = hlfir.minloc %[[VAL_1]]#0
! CHECK:  %[[VAL_3:.*]] = hlfir.shape_of %[[VAL_2]]
! CHECK:  %[[VAL_4:.*]]:3 = hlfir.associate %[[VAL_2]](%[[VAL_3]]) {adapt.valuebyref}
! CHECK:  fir.call @_QPtakes_array_arg(%[[VAL_4]]#1)
! CHECK:  hlfir.end_associate %[[VAL_4]]#1, %[[VAL_4]]#2 : !fir.ref<!fir.array<2xi32>>, i1
! CHECK:  hlfir.destroy %[[VAL_2]] : !hlfir.expr<2xi32>
