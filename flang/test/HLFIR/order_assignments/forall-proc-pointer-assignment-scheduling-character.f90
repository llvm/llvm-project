! Test analysis of character procedure pointer assignments inside FORALL.
! Character procedure gets their own tests because they are tracked differently
! in FIR because of the length of the function result.

! RUN: bbc -hlfir -o /dev/null -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" \
! RUN: --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only -I nw %s 2>&1 | FileCheck %s
! REQUIRES: asserts

module char_proc_ptr_forall
  type :: t
    procedure(f1), nopass, pointer :: p
  end type
contains
  pure character(2) function f1()
    f1 = "01"
  end function
  pure character(2) function f2()
    f2 = "02"
  end function
  pure character(2) function f3()
    f3 = "03"
  end function
  pure character(2) function f4()
    f4 = "04"
  end function
  pure character(2) function f5()
    f5 = "05"
  end function
  pure character(2) function f6()
    f6 = "06"
  end function
  pure character(2) function f7()
    f7 = "07"
  end function
  pure character(2) function f8()
    f8 = "08"
  end function
  pure character(2) function f9()
    f9 = "09"
  end function
  pure character(2) function f10()
    f10 = "10"
  end function

  integer pure function decode(c)
    character(2), intent(in) :: c
    decode = modulo(iachar(c(2:2))-49,10)+1 
  end function

  subroutine test_no_conflict(x)
    type(t) :: x(10)
    forall(i=1:10) x(i)%p => f1
  end subroutine
! CHECK: ------------ scheduling forall in _QMchar_proc_ptr_forallPtest_no_conflict ------------
! CHECK-NEXT: run 1 evaluate: forall/region_assign1

  subroutine test_need_to_save_rhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(i)%p => x(11-i)%p
  end subroutine
! CHECK: ------------ scheduling forall in _QMchar_proc_ptr_forallPtest_need_to_save_rhs ------------
! CHECK-NEXT: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

  subroutine test_need_to_save_lhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(decode(x(11-i)%p()))%p => f1
  end subroutine
! CHECK: ------------ scheduling forall in _QMchar_proc_ptr_forallPtest_need_to_save_lhs ------------
! CHECK: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1

  subroutine test_need_to_save_lhs_and_rhs(x)
    type(t) :: x(10)
    forall(i=1:10) x(decode(x(11-i)%p()))%p => x(modulo(-2*i, 11))%p
  end subroutine
! CHECK: ------------ scheduling forall in _QMchar_proc_ptr_forallPtest_need_to_save_lhs_and_rhs ------------
! CHECK: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
! CHECK: conflict: R/W
! CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
! CHECK-NEXT: run 2 evaluate: forall/region_assign1


! End-to-end test utilities for debugging purposes.

  subroutine reset(a)
    type(t) :: a(:)
    a = [t(f10), t(f9), t(f8), t(f7), t(f6), t(f5), t(f4), t(f3), t(f2), t(f1)]
  end subroutine

  subroutine print(a)
    type(t) :: a(:)
    print *, [(decode(a(i)%p()), i=1,10)]
  end subroutine

  logical function check_equal(a, expected)
    type(t) :: a(:)
    integer :: expected(:)
    check_equal = all([(decode(a(i)%p()), i=1,10)].eq.expected)
    if (.not.check_equal) then
      print *, "expected:", expected
      print *, "got:", [(decode(a(i)%p()), i=1,10)]
    end if
  end function
end module

! End-to-end test for debugging purposes (not verified by lit).
  use char_proc_ptr_forall
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
  print *, "PASS"
end
