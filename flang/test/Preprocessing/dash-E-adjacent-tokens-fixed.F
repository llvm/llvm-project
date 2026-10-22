! RUN: %flang -E %s 2>&1 | FileCheck %s
! RUN: %flang -E %s > %t.f
! RUN: %flang_fc1 -fdebug-unparse %t.f 2>&1 | FileCheck --strict-whitespace --check-prefix=UNPARSE %s
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck --strict-whitespace --check-prefix=UNPARSE %s

! Fixed form variant of dash-E-adjacent-tokens.F90: blanks that were
! removed by prescanning are restored, but not between the characters
! of a token formed by macro expansion.

#define D_OR_S() d
      subroutine s(n, a, i)
      integer n, i
      double precision a(n,n)
! Blanks the prescanner removed after names are restored, but not inside a
! character literal.
! UNPARSE: CHARACTER*8 hdr/"ab cd ef"/
      character*8 hdr /'ab cd ef'/
! CHECK: call dpotrf(1,
! UNPARSE: CALL dpotrf(1_4,n,a,n,i)
      call D_OR_S()potrf(1, n, a, n, i)
! CHECK: call   dgetrf(n,
! UNPARSE: CALL dgetrf(n,n,a,n,i)
      call   D_OR_S()getrf(n, n, a, n, i)
! CHECK: if (n .eq. 1) i = 0
! UNPARSE: IF (n==1_4){{ +}}i=0_4
      if (n .eq. 1) i = 0
! CHECK: !$omp parallel private(dx)
c$omp parallel private(D_OR_S()x)
c$omp end parallel
      end
