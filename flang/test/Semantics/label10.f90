! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: '60' not a FORMAT
! CHECK: data transfer use of '60'

subroutine s(a)
  real a(10)
  write(*,60) "Hi there"
60 continue
70 format (i8)
end subroutine s
