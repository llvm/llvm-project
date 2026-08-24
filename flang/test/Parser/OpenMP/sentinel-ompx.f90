! Implementation-defined extension sentinel "ompx" in free source form
! (OpenMP 5.2, section 3.1.2).  The !$ompx sentinel introduces implementation-
! defined extension directives and is recognized like the !$omp sentinel.

! RUN: %flang_fc1 -E -fopenmp %s 2>&1 | FileCheck %s --check-prefix=CHECK-E
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP

subroutine ompx_sub(a, b, c)
  real :: a, b, c
  !$ompx parallel &
  !$ompx private(a, b, c)
  a = b + c
  !$ompx end parallel
end subroutine

! The 5-character sentinel is preserved contiguously (not split as "!$omp x"),
! so the -E output round-trips.
! CHECK-E:{{^}}!$ompx parallel private(a, b, c)
! CHECK-E:{{^}}!$ompx end parallel

! With -fopenmp the extension directive is parsed as an OpenMP directive.
! CHECK-OMP: !$OMP PARALLEL PRIVATE(a,b,c)
! CHECK-OMP: !$OMP END PARALLEL

! Without -fopenmp the extension directive is ignored as a comment.
! CHECK-NO-OMP: SUBROUTINE ompx_sub
! CHECK-NO-OMP-NOT: !$OMP
