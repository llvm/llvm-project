!RUN: %flang_fc1 -fdebug-unparse -fopenacc %s | FileCheck %s
      subroutine s(a,b,n)
*$acc routine gang
      real a(n), b(n)
      integer(8) j
*$acc loop gang vector worker
      do 10 j = 1, n
*$acc atomic update
10    a(j) = a(j) + b(j)
      end

!CHECK: SUBROUTINE s (a, b, n)
!CHECK: !$ACC ROUTINE GANG
!CHECK:  REAL a(n), b(n)
!CHECK: !$ACC LOOP GANG VECTOR WORKER
!CHECK:  DO j=1_4,n
!CHECK: !$ACC ATOMIC UPDATE
!CHECK:   10  a(j)=a(j)+b(j)
!CHECK:  END DO
!CHECK: END SUBROUTINE
