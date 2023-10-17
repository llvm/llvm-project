! RUN: %python %S/test_errors.py %s %flang_fc1
! Don't expand scalars for allocatable components.
module m
  type t
    real, allocatable :: a(:)
  end type
  !ERROR: Scalar value cannot be expanded to shape of array component 'a'
  type(t) :: x = t(0.)
end module
