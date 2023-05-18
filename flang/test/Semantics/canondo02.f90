
! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: end do

SUBROUTINE sub00(a,b,n,m)
  INTEGER n,m
  REAL a(n,m), b(n,m)

  i = n-1
  DO 10 j = 1,m
     g = a(i,j) - b(i,j)
10   PRINT *, g
END SUBROUTINE sub00
