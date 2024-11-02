! RUN: %python %S/test_errors.py %s %flang_fc1
! test named constant declarations

function f1() result(x)
  !ERROR: A function result may not also be a named constant
  integer, parameter :: x = 1
end

