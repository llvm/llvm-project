! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -Wused-undefined-variable

integer function f()
  real, save :: c[*] = 0.
  real :: r
  r = c[1, stat=f]
end function

subroutine s
  real, save :: c[*] = 0.
  real :: r
  integer :: st
  r = c[1, stat=st]
  print *, r, st
end subroutine
