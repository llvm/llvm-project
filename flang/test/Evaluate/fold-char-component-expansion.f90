! RUN: %python %S/test_folding.py %s %flang_fc1
! A scalar CHARACTER value for an array component is repeated in every
! element when the structure constructor is folded.
module m
  type t1
    character(3) c(2, 3)
  end type
  type t2
    character(kind=4, len=2) c(4)
  end type
  type t3
    character(0) c(5)
  end type
  type(t1), parameter :: x1 = t1('abc')
  type(t2), parameter :: x2 = t2(4_'xy')
  type(t3), parameter :: x3 = t3('')
  logical, parameter :: test_x1 = all(x1%c == 'abc') .and. size(x1%c) == 6
  logical, parameter :: test_x1_elem = x1%c(2, 3) == 'abc'
  logical, parameter :: test_x2 = all(x2%c == 4_'xy') .and. len(x2%c) == 2
  logical, parameter :: test_x3 = size(x3%c) == 5 .and. len(x3%c) == 0
end
