! Test analysis of polymorphic pointer assignment inside FORALL.
! The analysis must detect if the evaluation of the LHS or RHS may be impacted
! by the pointer assignments, or if the forall can be lowered into a single
! loop without any temporary copy.

! RUN: bbc -hlfir -o /dev/null -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN: --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts
module forall_poly_pointers
  type base
    integer :: i
  end type
  type, extends(base) :: extension
    integer :: j
  end type
  type ptr_wrapper
    class(base), pointer :: p
  end type
contains

! Simple case that can be lowered into a single loop.
subroutine test_no_conflict(n, a, somet)
 integer :: n
 type(ptr_wrapper) :: a(:)
 class(base), target :: somet
 forall(i=1:n) a(i)%p => somet
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_poly_pointersPtest_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine test_no_conflict2(n, a, somet)
 integer :: n
 type(ptr_wrapper) :: a(:)
 type(base), target :: somet
 forall(i=1:n) a(i)%p => somet
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_poly_pointersPtest_no_conflict2 ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine test_rhs_conflict(n, a)
 integer :: n
 type(ptr_wrapper) :: a(:)
 forall(i=1:n) a(i)%p => a(n+1-i)%p
end subroutine
! CHECK: ------------ scheduling forall in _QMforall_poly_pointersPtest_rhs_conflict ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1
end module

! End to end test provided for debugging purpose (not run by lit).
program end_to_end
  use forall_poly_pointers
  integer, parameter :: n = 10
  type(extension), target, save :: data(n) = [(extension(i, 100+i), i=1,n)]
  type(ptr_wrapper) :: pointers(n)
  ! Print pointer/target mapping baseline.
  call reset_pointers(pointers)
  if (.not.check_equal(pointers, [10,9,8,7,6,5,4,3,2,1])) stop 1
  if (.not.check_type(pointers, [(modulo(i,3).eq.0, i=1,n)])) stop 2

  ! Test dynamic type is correctly set.
  call test_no_conflict(n, pointers, data(1))
  if (.not.check_equal(pointers, [(1,i=1,10)])) stop 3
  if (.not.check_type(pointers, [(.true.,i=1,10)])) stop 4
  call test_no_conflict(n, pointers, data(1)%base)
  if (.not.check_equal(pointers, [(1,i=1,10)])) stop 5
  if (.not.check_type(pointers, [(.false.,i=1,10)])) stop 6

  call test_no_conflict2(n, pointers, data(1)%base)
  if (.not.check_equal(pointers, [(1,i=1,10)])) stop 7
  if (.not.check_type(pointers, [(.false.,i=1,10)])) stop 8

  ! Test RHS conflict.
  call reset_pointers(pointers)
  call test_rhs_conflict(n, pointers)
  if (.not.check_equal(pointers, [(i, i=1,10)])) stop 9
  if (.not.check_type(pointers, [(modulo(i,3).eq.2, i=1,n)])) stop 10

  print *, "PASS"
contains
subroutine reset_pointers(a)
  type(ptr_wrapper) :: a(:)
  do i=1,n
    if (modulo(i,3).eq.0) then
      a(i)%p => data(n+1-i)
    else
      a(i)%p => data(n+1-i)%base
    end if
  end do
end subroutine
logical function check_equal(a, expected)
  type(ptr_wrapper) :: a(:)
  integer :: expected(:)
  check_equal = all([(a(i)%p%i, i=1,10)].eq.expected)
  if (.not.check_equal) then
    print *, "expected:", expected
    print *, "got:", [(a(i)%p%i, i=1,10)]
  end if
end function
logical function check_type(a, expected)
  type(ptr_wrapper) :: a(:)
  logical :: expected(:)
  check_type = all([(same_type_as(a(i)%p, extension(1,1)), i=1,10)].eqv.expected)
  if (.not.check_type) then
    print *, "expected:", expected
    print *, "got:", [(same_type_as(a(i)%p, extension(1,1)), i=1,10)]
  end if
end function
end
