! RUN: %flang -E %s 2>&1 | FileCheck %s
! RUN: %flang -E %s > %t.f90
! RUN: %flang_fc1 -fdebug-unparse %t.f90 2>&1 | FileCheck --check-prefix=UNPARSE %s
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck --check-prefix=UNPARSE %s

! -E output must not insert blanks between characters that are adjacent
! in the cooked character stream, so that compiling the -E output gives
! the same result as compiling the original source.

#define D_OR_S() d
#define PREFIX(x) x

subroutine s(n, a, i)
  integer n, i
  double precision a(n,n)
! CHECK: call dpotrf('L', n, a, n, i)
! UNPARSE: CALL dpotrf("L",n,a,n,i)
  call D_OR_S()potrf('L', n, a, n, i)
! CHECK: call dpotrs('L', n, a, n, i)
! UNPARSE: CALL dpotrs("L",n,a,n,i)
  call PREFIX(d)potrs('L', n, a, n, i)
! CHECK: call dgetrf(n, n, a, n, i)
! UNPARSE: CALL dgetrf(n,n,a,n,i)
  call   D_OR_S()getrf(n, n, a, n, i)
! CHECK: call dpotri('L', n, a, n, i)
! UNPARSE: CALL dpotri("L",n,a,n,i)
#include "dash-E-adjacent-tokens.inc"
! CHECK: call dtrtri('L', n, a, n, i)
! UNPARSE: CALL dtrtri("L",n,a,n,i)
	call D_OR_S()trtri('L', n, a, n, i)
end subroutine
