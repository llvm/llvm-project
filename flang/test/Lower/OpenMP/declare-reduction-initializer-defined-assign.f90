! Test user-defined assignment in declare reduction initializer.
! Verifies that `initializer(omp_priv = t(1))` correctly dispatches to the
! user-defined `assignment(=)` subroutine, not intrinsic assignment.
!
! This is a regression test for https://github.com/llvm/llvm-project/issues/184927

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m_defined_assign
  implicit none
  type :: t
     integer :: val = -999
  end type t

  interface assignment(=)
     module procedure custom_assign
  end interface

contains
  subroutine custom_assign(lhs, rhs)
    type(t), intent(out) :: lhs
    type(t), intent(in)  :: rhs
    lhs%val = rhs%val * 10
  end subroutine
end module

subroutine test_defined_assign_init()
  use m_defined_assign
  implicit none
  integer :: i
  type(t) :: x

  !$omp declare reduction(add_t : t : omp_out%val = omp_out%val + omp_in%val) &
  !$omp&   initializer(omp_priv = t(1))

  x = t(0)
  !$omp parallel do reduction(add_t : x) num_threads(2)
  do i = 1, 2
     x%val = x%val + 1
  end do
  !$omp end parallel do
end subroutine

!CHECK: omp.declare_reduction @add_t :
!CHECK-SAME: alloc {
!CHECK:   %[[ALLOCA:.*]] = fir.alloca
!CHECK:   omp.yield(%[[ALLOCA]] :
!CHECK: } init {
!CHECK: ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<!fir.type<_QMm_defined_assignTt{val:i32}>>,
!CHECK-SAME: %[[INIT_ARG1:.*]]: !fir.ref<!fir.type<_QMm_defined_assignTt{val:i32}>>):
!CHECK:   %[[OMP_ORIG:.*]]:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"}
!CHECK:   %[[OMP_PRIV:.*]]:2 = hlfir.declare %[[INIT_ARG1]] {uniq_name = "omp_priv"}
!CHECK:   %[[INIT_ADDR:.*]] = fir.address_of(@_QQro._QMm_defined_assignTt.0)
!CHECK:   %[[INIT_DECL:.*]]:2 = hlfir.declare %[[INIT_ADDR]]
!CHECK:   %[[AS_EXPR:.*]] = hlfir.as_expr %[[INIT_DECL]]#0
!CHECK:   %[[ASSOC:.*]]:3 = hlfir.associate %[[AS_EXPR]] {adapt.valuebyref}
!CHECK:   fir.call @_QMm_defined_assignPcustom_assign(%[[OMP_PRIV]]#0, %[[ASSOC]]#0)
!CHECK:   hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2
!CHECK:   omp.yield(%[[INIT_ARG1]] :
!CHECK: } combiner {
