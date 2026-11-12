! RUN: %python %S/test_errors.py %s %flang_fc1
type :: t
  integer :: n
end type
type(t) :: x
real, target, save :: a(1)
real, parameter :: arrparm(1) = [3.14159]
real, pointer :: p
real :: y
data x/t(1)/
data p/a(1)/
!ERROR: DATA statement value initializes 'y' with an array
data y/arrparm/
end
