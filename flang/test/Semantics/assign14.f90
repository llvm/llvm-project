! RUN: %python %S/test_errors.py %s %flang_fc1
! Can't associate a pointer with a substring of a character literal
program main
  character(:), pointer :: cp
  !ERROR: Target associated with pointer 'cp' must be a designator or a call to a pointer-valued function
  cp => "abcd"(1:4)
end
