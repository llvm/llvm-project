! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SAME_TYPE_AS() and EXTENDS_TYPE_OF()
module m

  type :: t1
    real :: x
  end type
  type :: t2(k)
    integer, kind :: k
    real(kind=k) :: x
  end type
  type :: t3
    real :: x
  end type
  type, extends(t1) :: t4
    integer :: y
  end type

  type(t1) :: x1, y1
  type(t2(4)) :: x24, y24
  type(t2(8)) :: x28
  type(t3) :: x3
  type(t4) :: x4
  class(t1), allocatable :: a1
  class(t3), allocatable :: a3

  logical, parameter :: test_1 = same_type_as(x1, x1)
  logical, parameter :: test_2 = same_type_as(x1, y1)
  logical, parameter :: test_3 = same_type_as(x24, x24)
  logical, parameter :: test_4 = same_type_as(x24, y24)
  logical, parameter :: test_5 = same_type_as(x24, x28) ! ignores parameter
  logical, parameter :: test_6 = .not. same_type_as(x1, x3)
  logical, parameter :: test_7 = .not. same_type_as(a1, a3)

  logical, parameter :: test_11 = extends_type_of(x1, y1)
  logical, parameter :: test_12 = extends_type_of(x24, x24)
  logical, parameter :: test_13 = extends_type_of(x24, y24)
  logical, parameter :: test_14 = extends_type_of(x24, x28) ! ignores parameter
  logical, parameter :: test_15 = .not. extends_type_of(x1, x3)
  logical, parameter :: test_16 = .not. extends_type_of(a1, a3)
  logical, parameter :: test_17 = .not. extends_type_of(x1, x4)
  logical, parameter :: test_18 = extends_type_of(x4, x1)
end module
