! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of complex components
module m
  complex, parameter :: z = (1., 2.)
  logical, parameter :: test_1 = z%re == 1.
  logical, parameter :: test_2 = z%im == 2.
  logical, parameter :: test_3 = real(z+z) == 2.
  logical, parameter :: test_4 = aimag(z+z) == 4.
  type :: t
    complex :: z
  end type
  type(t), parameter :: tz(*) = [t((3., 4.)), t((5., 6.))]
  logical, parameter :: test_5 = all(tz%z%re == [3., 5.])
  logical, parameter :: test_6 = all(tz%z%im == [4., 6.])
end module
