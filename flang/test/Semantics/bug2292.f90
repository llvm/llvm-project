!RUN: %python %S/test_errors.py %s %flang_fc1
subroutine sub(lb)
  !ERROR: The lower bounds of the parameter 'const' are not constant
  integer, parameter :: const(lb:*) = [0]
end
