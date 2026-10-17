! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  !ERROR: Module 'm' cannot USE itself
  use m
end module m