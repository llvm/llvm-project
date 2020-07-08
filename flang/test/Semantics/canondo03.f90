
! RUN: %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: 10 continue
! CHECK: end do

SUBROUTINE sub00(a,b,n,m)
  INTEGER n,m
  REAL a(n,m), b(n,m)

  i = n-1
  DO 10 j = 1,m
     g = a(i,j) - b(i,j)
     PRINT *, g
10 END DO
END SUBROUTINE sub00
