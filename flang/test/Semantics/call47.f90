! RUN: %python %S/test_errors.py %s %flang_fc1
! Test F2018 15.5.1 C1535 - keyword arguments are not allowed when the
! interface is implicit. Statement functions have implicit interfaces.

program test_stmt_func_keyword
  integer :: f1, f2, x, y, c

  ! Statement function definitions
  f1(x) = x / 2
  f2(x, y) = x + y

  ! Calling a statement function with a keyword argument is not allowed
  ! because statement functions have implicit interfaces.
  !ERROR: Keyword 'x=' may not appear in a reference to a procedure with an implicit interface
  c = f1(x=10)

  ! Wrong keyword name - gets both implicit interface error and unrecognized keyword error
  !ERROR: Keyword 'y=' may not appear in a reference to a procedure with an implicit interface
  !ERROR: Argument keyword 'y=' is not recognized for this procedure reference
  c = f1(y=10)

  ! Two keyword arguments - each gets its own diagnostic
  !ERROR: Keyword 'x=' may not appear in a reference to a procedure with an implicit interface
  !ERROR: Keyword 'y=' may not appear in a reference to a procedure with an implicit interface
  c = f2(x=10, y=20)
end program

! A parameterized derived type actual argument to a statement function does not
! require an explicit interface, so the reference below is valid (no error).
subroutine pdt_actual_to_stmt_func()
  type t(k)
    integer, kind :: k = 1
  end type

  type(t) :: x
  integer :: f

  f(x) = 0
  print *, f(x)
end subroutine
