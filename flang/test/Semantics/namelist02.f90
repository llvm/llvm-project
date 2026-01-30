! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

program p
  implicit none
  integer, parameter :: k = 3
  !WARNING: A namelist group object 'k' must not be a PARAMETER
  namelist /g/ k
end program
