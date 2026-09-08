! RUN: %python %S/test_errors.py %s %flang_fc1
program main
  type foo
  end type foo
  !ERROR: 'foo' is already declared in this scoping unit
  integer foo(1) /2/
end program main
