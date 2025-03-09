! Test analysis of pointer assignment inside FORALL.
! The analysis must detect if the evaluation of the LHS or RHS may be impacted
! by the pointer assignments, or if the forall can be lowered into a single
! loop without any temporary copy.

! RUN: bbc -hlfir -o /dev/null -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN: --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts
module forall_pointers
  type t
    integer :: i
  end type
  type ptr_wrapper
    type(t), pointer :: p
  end type
contains

! Simple case that can be lowered into a single loop.
subroutine test_no_conflict(n, a, somet)
 integer :: n
 type(ptr_wrapper), allocatable :: a(:)
 type(t), target :: somet
 forall(i=1:n) a(i)%p => somet
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointersPtest_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

! Case where the pointer target evaluations are impacted by the pointer
! assignments and should be evaluated for each iteration before doing
! any pointer assignment.
! The test is transposing an array of (wrapped) pointers.
subroutine test_need_to_save_rhs(n, a)
 integer :: n
 type(ptr_wrapper) :: a(:)
 forall(i=1:n) a(i)%p => a(n+1-i)%p
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointersPtest_need_to_save_rhs ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

! Case where the pointer descriptor address evaluations are impacted by the
! assignments and should be evaluated for each iteration before doing
! any pointer assignment.
subroutine test_need_to_save_lhs(n, a, somet)
 integer :: n
 type(ptr_wrapper) :: a(:)
 type(t), target :: somet
 forall(i=1:n) a(a(n+1-i)%p%i)%p => somet
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointersPtest_need_to_save_lhs ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

! Case where both the computation of the target and descriptor addresses are
! impacted by the assignment and need to be all evaluated before doing any
! assignment.
subroutine test_need_to_save_lhs_and_rhs(n, a)
 integer :: n
 type(ptr_wrapper) :: a(:)
 forall(i=1:n) a(a(n+1-i)%p%i)%p => a(modulo(-2*i, n+1))%p
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointersPtest_need_to_save_lhs_and_rhs ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1
end module

! End to end test provided for debugging purpose (not run by lit).
program end_to_end
  use forall_pointers
  integer, parameter :: n = 10
  type(t), target, save :: data(n) = [(t(i), i=1,n)]
  type(ptr_wrapper) :: pointers(n)
  ! Print pointer/target mapping baseline.
  ! Expect: 10 9 8 7 6 5 4 3 2 1
  call reset_pointers(pointers)
  call print_pointers(pointers)

  ! Test case where RHS target addresses must be saved in FORALL.
  ! Expect: 1 2 3 4 5 6 7 8 9 10
  call test_need_to_save_rhs(n, pointers)
  call print_pointers(pointers)

  ! Test case where LHS pointer addresses must be saved in FORALL.
  ! Expect: 1 1 1 1 1 1 1 1 1 1
  call reset_pointers(pointers)
  call test_need_to_save_lhs(n, pointers, data(1))
  call print_pointers(pointers)

  ! Test case where bot RHS target addresses and LHS pointer addresses must be
  ! saved in FORALL.
  ! Expect: 2 4 6 8 10 1 3 5 7 9
  call reset_pointers(pointers)
  call test_need_to_save_lhs_and_rhs(n, pointers)
  call print_pointers(pointers)
contains
subroutine reset_pointers(a)
  type(ptr_wrapper) :: a(:)
  do i=1,n
    a(i)%p => data(n+1-i)
  end do
end subroutine
subroutine print_pointers(a)
  type(ptr_wrapper) :: a(:)
  print *, [(a(i)%p%i, i=lbound(a,1), ubound(a,1))]
end subroutine
end
