! RUN: %S/test_any.sh %s %t %f18
! negative test -- invalid labels, out of range

! EXEC: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: end do

SUBROUTINE sub00(a,b,n,m)
  INTEGER n,m
  REAL a(n,m), b(n,m)

  DO 10 j = 1,m
     DO 10 i = 1,n
        g = a(i,j) - b(i,j)
10      PRINT *, g
END SUBROUTINE sub00
