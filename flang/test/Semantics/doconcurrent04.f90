! RUN: %S/test_any.sh %s %t %f18
! C1122 The index-name shall be a named scalar variable of type integer.
! EXEC: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: Must have INTEGER type, but is REAL\\(4\\)

subroutine do_concurrent_test1(n)
  implicit none
  integer :: n
  real :: j
  do 20 concurrent (j = 1:n)
20 enddo
end subroutine do_concurrent_test1
