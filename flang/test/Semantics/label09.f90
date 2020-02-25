! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: label '60' was not found

subroutine s(a)
  real a(10)
  write(*,60) "Hi there"
end subroutine s
