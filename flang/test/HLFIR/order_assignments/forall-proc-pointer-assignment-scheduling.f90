! Test analysis of procedure pointer assignments inside FORALL.

! RUN: bbc -hlfir -o /dev/null -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN: --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only -I nw %s 2>&1 | FileCheck %s
! REQUIRES: asserts

module proc_ptr_forall
  type :: t
    procedure(f1), nopass, pointer :: p
  end type
contains
  pure integer function f1()
    f1 = 1
  end function
  pure integer function f2()
    f2 = 2
  end function
  pure integer function f3()
    f3 = 3
  end function
  pure integer function f4()
    f4 = 4
  end function
  pure integer function f5()
    f5 = 5
  end function
  pure integer function f6()
    f6 = 6
  end function
  pure integer function f7()
    f7 = 7
  end function
  pure integer function f8()
    f8 = 8
  end function
  pure integer function f9()
    f9 = 9
  end function
  pure integer function f10()
    f10 = 10
  end function

  subroutine test_no_conflict(x)
    type(t) :: x(10)
    forall(i=1:10) x(i)%p => f1
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

  subroutine test_need_to_save_rhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(i)%p => x(11-i)%p
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_need_to_save_rhs ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

  subroutine test_need_to_save_lhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(x(11-i)%p())%p => f1
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_need_to_save_lhs ------------
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

  subroutine test_need_to_save_lhs_and_rhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(x(11-i)%p())%p => x(modulo(-2*i, 11))%p
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_need_to_save_lhs_and_rhs ------------
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

  subroutine test_null_no_conflict(x)
    type(t) :: x(10)
    forall(i=1:10) x(i)%p => null()
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_null_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

  subroutine test_null_need_to_save_lhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(x(11-i)%p())%p => null()
  end subroutine
! CHECK: ------------ scheduling forall in _QMproc_ptr_forallPtest_null_need_to_save_lhs ------------
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: unknown effect: %{{.*}} = fir.call
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

! End-to-end test utilities for debugging purposes.

  subroutine reset(a)
    type(t) :: a(:)
    a = [t(f10), t(f9), t(f8), t(f7), t(f6), t(f5), t(f4), t(f3), t(f2), t(f1)]
  end subroutine

  subroutine print(a)
    type(t) :: a(:)
    print *, [(a(i)%p(), i=1,10)]
  end subroutine

  logical function check_equal(a, expected)
    type(t) :: a(:)
    integer :: expected(:)
    check_equal = all([(a(i)%p(), i=1,10)].eq.expected)
    if (.not.check_equal) then
      print *, "expected:", expected
      print *, "got:", [(a(i)%p(), i=1,10)]
    end if
  end function

  logical function check_association(a, expected)
    type(t) :: a(:)
    logical :: expected(:)
    check_association = all([(associated(a(i)%p), i=1,10)].eqv.expected)
    if (.not.check_association) then
      print *, "expected:", expected
      print *, "got:", [(associated(a(i)%p), i=1,10)]
    end if
  end function

end module

! End-to-end test for debugging purposes (not verified by lit).
  use proc_ptr_forall
  type(t) :: a(10)

  call reset(a)
  call test_need_to_save_rhs(a)
  if (.not.check_equal(a, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) stop 1

  call reset(a)
  call test_need_to_save_lhs(a)
  if (.not.check_equal(a, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) stop 2

  call reset(a)
  call test_need_to_save_lhs_and_rhs(a)
  if (.not.check_equal(a, [2, 4, 6, 8, 10, 1, 3, 5, 7, 9])) stop 3

  call reset(a)
  call test_null_need_to_save_lhs(a)
  if (.not.check_association(a, [(.false., i=1,10)])) stop 4

  print *, "PASS"
end
