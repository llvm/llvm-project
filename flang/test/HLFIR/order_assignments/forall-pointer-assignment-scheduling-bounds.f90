! Test analysis of pointer assignment inside FORALL with lower bounds or bounds
! remapping.
! The analysis must detect if the evaluation of the LHS or RHS may be impacted
! by the pointer assignments, or if the forall can be lowered into a single
! loop without any temporary copy.

! RUN: bbc -hlfir -o /dev/null -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN: --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts
module forall_pointers_bounds
  type ptr_wrapper
    integer, pointer :: p(:, :)
  end type
contains

! Simple case that can be lowered into a single loop.
subroutine test_lb_no_conflict(a, iarray)
 type(ptr_wrapper) :: a(:)
 integer, target :: iarray(:, :)
 forall(i=lbound(a,1):ubound(a,1)) a(i)%p(2*(i-1)+1:,2*i:) => iarray
end subroutine

subroutine test_remapping_no_conflict(a, iarray)
 type(ptr_wrapper) :: a(:)
 integer, target :: iarray(6)
 ! Reshaping 6 to 2x3 with custom lower bounds.
 forall(i=lbound(a,1):ubound(a,1)) a(i)%p(2*(i-1)+1:2*(i-1)+2,2*i:2*i+2) => iarray
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointers_boundsPtest_remapping_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

! Bounds expression conflict. Note that even though they are syntactically on
! the LHS,they are saved with the RHS because they are applied when preparing the
! new descriptor value pointing to the RHS.
subroutine test_lb_conflict(a, iarray)
 type(ptr_wrapper) :: a(:)
 integer, target :: iarray(:, :)
 integer :: n
 n = ubound(a,1)
 forall(i=lbound(a,1):ubound(a,1)) a(i)%p(a(n+1-i)%p(1,1):,a(n+1-i)%p(2,1):) => iarray
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_pointers_boundsPtest_lb_conflict ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

end module

! End to end test provided for debugging purpose (not run by lit).
program end_to_end
  use forall_pointers_bounds
  integer, parameter :: n = 5
  integer, target, save :: data(2, 2, n) = reshape([(i, i=1,size(data))], shape=shape(data))
  integer, target, save :: data2(6) = reshape([(i, i=1,size(data2))], shape=shape(data2))
  type(ptr_wrapper) :: pointers(n)
  ! Print pointer/target mapping baseline.
  call reset_pointers(pointers)
  if (.not.check_equal(pointers, [17,18,19,20,13,14,15,16,9,10,11,12,5,6,7,8,1,2,3,4])) stop 1

  call reset_pointers(pointers)
  call test_lb_no_conflict(pointers, data(:, :, 1))
  if (.not.check_equal(pointers, [([1,2,3,4],i=1,n)])) stop 2
  if (.not.all([(lbound(pointers(i)%p), i=1,n)].eq.[(i, i=1,2*n)])) stop 3

  call reset_pointers(pointers)
  call test_remapping_no_conflict(pointers, data2)
  if (.not.check_equal(pointers, [([1,2,3,4,5,6],i=1,n)])) stop 4
  if (.not.all([(lbound(pointers(i)%p), i=1,n)].eq.[(i, i=1,2*n)])) stop 5
  if (.not.all([(ubound(pointers(i)%p), i=1,n)].eq.[([2*(i-1)+2, 2*i+2], i=1,n)])) stop 6

  call reset_pointers(pointers)
  call test_lb_conflict(pointers, data(:, :, 1))
  if (.not.check_equal(pointers, [([1,2,3,4],i=1,n)])) stop 7
  if (.not.all([(lbound(pointers(i)%p), i=1,n)].eq.[([data(1,1,i), data(2,1,i)], i=1,n)])) stop 8

  print *, "PASS"
contains
subroutine reset_pointers(a)
  type(ptr_wrapper) :: a(:)
  do i=1,n
    a(i)%p => data(:, :, n+1-i)
  end do
end subroutine
logical function check_equal(a, expected)
  type(ptr_wrapper) :: a(:)
  integer :: expected(:)
  check_equal = all([(a(i)%p, i=1,n)].eq.expected)
  if (.not.check_equal) then
    print *, "expected:", expected
    print *, "got:", [(a(i)%p, i=1,n)]
  end if
end function
end
