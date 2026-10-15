! RUN: %python %S/test_errors.py %s %flang_fc1

! F2023 15.8 p2: A SIMPLE procedure is also PURE.
! F2023 C15102: An internal subprogram of a pure subprogram must be pure

! A SIMPLE subprogram may not contain an impure internal subprogram
! C15113 additionally requires the internal subprogram to be SIMPLE
simple subroutine s(x)
  integer, intent(inout) :: x
  x = x + 1
contains
  !ERROR: An internal subprogram of a pure subprogram must also be pure
  subroutine inner(y)
    integer, intent(inout) :: y
    y = y + 1
  end subroutine
end subroutine

! A SIMPLE internal subprogram satisfies the PURE requirement
pure subroutine t(x)
  integer, intent(inout) :: x
  call inner(x)
contains
  simple subroutine inner(y)
    integer, intent(inout) :: y
    y = y + 1
  end subroutine
end subroutine