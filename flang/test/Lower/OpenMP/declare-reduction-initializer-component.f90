! Test component-level initializer in declare reduction for derived types.
! Verifies that `initializer(omp_priv%member = 0)` correctly lowers to
! a component designate + assign (hlfir.designate + hlfir.assign), rather
! than storing the scalar directly to the whole derived-type reference.
!
! This is a regression test for https://github.com/llvm/llvm-project/issues/184927

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine test_component_init()
  implicit none
  type :: t
     integer :: member
  end type t
  integer :: i
  !$omp declare reduction(add_member : t : &
  !$omp&   omp_out%member = omp_out%member + omp_in%member) &
  !$omp&   initializer(omp_priv%member = 0)
  type(t) :: x
  x%member = 0
  !$omp parallel do reduction(add_member : x) num_threads(2)
  do i = 1, 10
     x%member = x%member + 1
  end do
  !$omp end parallel do
end subroutine

!CHECK: omp.declare_reduction @add_member : !fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>
!CHECK-SAME: alloc {
!CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.type<_QFtest_component_initTt{member:i32}>
!CHECK:   omp.yield(%[[ALLOCA]] : !fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>)
!CHECK: } init {
!CHECK: ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>,
!CHECK-SAME: %[[INIT_ARG1:.*]]: !fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>):
!CHECK:   %[[OMP_ORIG:.*]]:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"}
!CHECK:   %[[OMP_PRIV:.*]]:2 = hlfir.declare %[[INIT_ARG1]] {uniq_name = "omp_priv"}
!CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK:   %[[MEMBER:.*]] = hlfir.designate %[[OMP_PRIV]]#0{"member"} : (!fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>) -> !fir.ref<i32>
!CHECK:   hlfir.assign %[[ZERO]] to %[[MEMBER]] : i32, !fir.ref<i32>
!CHECK:   omp.yield(%[[INIT_ARG1]] : !fir.ref<!fir.type<_QFtest_component_initTt{member:i32}>>)
!CHECK: } combiner {
