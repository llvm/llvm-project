
! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: Label '0' is out of range
! CHECK: Label '100000' is out of range
! CHECK: Label '123456' is out of range
! CHECK: Label '123456' was not found
! CHECK: Label '1000' is not distinct

subroutine sub00(a,b,n,m)
  real a(n)
  real :: b(m)
0 print *, "error"
100000 print *, n
  goto 123456
1000 print *, m
1000 print *, m+1
end subroutine sub00
