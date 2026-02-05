! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  use iso_c_binding
  !ERROR: CDEFINED variable cannot be initialized
  integer(c_int), bind(C, name='c_global', CDEFINED) :: c  = 42
end
