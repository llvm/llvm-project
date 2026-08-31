! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  integer :: t
  parameter(i=1)
  !ERROR: 'i' may not have both the POINTER and PARAMETER attributes
  integer, pointer :: i => t
end module
