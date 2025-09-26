! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
module m
  use iso_c_binding
  !WARNING: CDEFINED variable should not have an initializer [-Wcdefined-init]
  integer(c_int), bind(C, name='c_global', CDEFINED) :: c  = 42
end
