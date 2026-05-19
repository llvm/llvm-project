! RUN: %python %S/test_errors.py %s %flang_fc1
! Test F'2018 15.5.1 C1535 - keyword arguments are not allowed when the
! interface is implicit. Statement functions have implicit interfaces.

program test_stmt_func_keyword
  integer :: f1, x, c

  ! Statement function definition
  f1(x) = x / 2

  ! Calling a statement function with a keyword argument is not allowed
  ! because statement functions have implicit interfaces.
  !ERROR: Keyword 'x=' may not appear in a reference to a procedure with an implicit interface
  c = f1(x=10)
end program
