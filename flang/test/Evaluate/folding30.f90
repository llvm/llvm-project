! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of structure constructors in array constructors
module m
  type :: t1
    integer :: n
  end type
  type(t1), parameter :: xs1(*) = [(t1(j),j=1,5,2)]
  type(t1), parameter :: xs2(*) = [(t1(j),j=5,1,-2)]
  logical, parameter :: test_1 = all(xs1%n == [1, 3, 5])
  logical, parameter :: test_2 = all(xs2%n == [5, 3, 1])
end module
