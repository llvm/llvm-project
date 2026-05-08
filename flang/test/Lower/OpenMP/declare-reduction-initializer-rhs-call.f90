! Test that a function call on the RHS of a component-level initializer in
! declare reduction is correctly lowered through the assignment path (not
! the ProcedureRef/subroutine path). Verifies that the init region contains
! a call to the function, followed by a component designate and assign.
!
! This is a regression test for https://github.com/llvm/llvm-project/issues/184927

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  implicit none
  type :: t
     integer :: member
  end type t
contains
  function init_val() result(res)
    integer :: res
    res = 42
  end function
end module

subroutine test_rhs_call()
  use m
  implicit none
  integer :: i
  type(t) :: x

  !$omp declare reduction(add_t : t : omp_out%member = omp_out%member + omp_in%member) &
  !$omp&   initializer(omp_priv%member = init_val())

  x%member = 0
  !$omp parallel do reduction(add_t : x) num_threads(2)
  do i = 1, 2
     x%member = x%member + 1
  end do
  !$omp end parallel do
end subroutine

!CHECK: omp.declare_reduction @add_t :
!CHECK-SAME: alloc {
!CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.type<_QMmTt{member:i32}>
!CHECK:   omp.yield(%[[ALLOCA]] :
!CHECK: } init {
!CHECK: ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<!fir.type<_QMmTt{member:i32}>>,
!CHECK-SAME: %[[INIT_ARG1:.*]]: !fir.ref<!fir.type<_QMmTt{member:i32}>>):
!CHECK:   %[[OMP_ORIG:.*]]:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"}
!CHECK:   %[[OMP_PRIV:.*]]:2 = hlfir.declare %[[INIT_ARG1]] {uniq_name = "omp_priv"}
!CHECK:   %[[CALL:.*]] = fir.call @_QMmPinit_val() {{.*}} : () -> i32
!CHECK:   %[[MEMBER:.*]] = hlfir.designate %[[OMP_PRIV]]#0{"member"} : (!fir.ref<!fir.type<_QMmTt{member:i32}>>) -> !fir.ref<i32>
!CHECK:   hlfir.assign %[[CALL]] to %[[MEMBER]] : i32, !fir.ref<i32>
!CHECK:   omp.yield(%[[INIT_ARG1]] : !fir.ref<!fir.type<_QMmTt{member:i32}>>)
!CHECK: } combiner {
