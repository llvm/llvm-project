! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
module m
  use iso_c_binding
  !WARNING: CDEFINED variable cannot be initialized [-Wcdefined-init]
  integer(c_int), bind(C, name='c_global', CDEFINED) :: c  = 42
end
