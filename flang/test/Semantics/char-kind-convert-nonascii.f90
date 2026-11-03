! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Conversion of CHARACTER values between kinds folds only when every
! character is 7-bit ASCII; otherwise the value is not a constant.
module m
  type t
    ! CHECK: error: Initialization expression for 'c' ({{.*}}) cannot be computed as a constant value
    character(kind=4, len=3) :: c = 'é!'
  end type
  type u
    character(kind=4, len=3) :: c
  end type
  ! CHECK: error: Value of named constant 'p1' ({{.*}}) cannot be computed as a constant value
  character(kind=4, len=3), parameter :: p1 = 'é!'
  ! CHECK: error: Value of named constant 'p2' ({{.*}}) cannot be computed as a constant value
  character(kind=1, len=3), parameter :: p2 = 4_'é!'
  ! CHECK: error: Value of named constant 'p3' ({{.*}}) cannot be computed as a constant value
  character(kind=2, len=3), parameter :: p3 = 4_'é!'
  ! CHECK: error: Value of named constant 'p4' ({{.*}}) cannot be computed as a constant value
  character(kind=4, len=3), parameter :: p4 = 2_'é!'
  character(kind=4, len=3) :: d1
  ! CHECK: error: DATA statement value '{{.*}}' for 'd1' is not a constant
  data d1 /'é!'/
  ! CHECK: error: Value of named constant 's1' ({{.*}}) cannot be computed as a constant value
  type(u), parameter :: s1 = u('é!')
  ! CHECK: error: Value of named constant 'a1' ({{.*}}) cannot be computed as a constant value
  character(kind=4, len=3), parameter :: a1(1) = &
      [character(kind=4, len=3) :: 'é!']
  ! CHECK-NOT: error:
  character(kind=4, len=3), parameter :: ok1 = 'ab!'
  character(kind=1, len=3), parameter :: ok2 = 4_'ab!'
  character(kind=2, len=3), parameter :: ok3 = 4_''
  type(u), parameter :: ok4 = u('ab!')
  logical, parameter :: test_ok = ok1 == 4_'ab!' .and. ok2 == 'ab!' &
      .and. ok3 == 2_'   ' .and. ok4%c == 4_'ab!'
  integer, parameter :: check_ok = merge(1, 0, test_ok) / merge(1, 0, test_ok)
end module
