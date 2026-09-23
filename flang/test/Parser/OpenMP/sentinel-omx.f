! Implementation-defined extension sentinel "omx" in fixed source form
! (OpenMP 5.2, section 3.1.1).  The sentinels !$omx, c$omx and *$omx introduce
! implementation-defined extension directives and are recognized like !$omp;
! the prescanner normalizes the fixed-form comment character (c, *) to '!'.

! RUN: %flang_fc1 -E -fopenmp %s 2>&1 | FileCheck %s --check-prefix=CHECK-E
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP

      subroutine omx_forms(a, n)
        integer :: n
        real :: a(n)
!$omx parallel
c$omx barrier
        call work(a, n)
*$omx end parallel
      end subroutine

! A continuation line for an omx directive uses a non-blank in column 6.
      subroutine omx_cont(a, b, c)
        real :: a, b, c
c$omx parallel
c$omx+ private(a, b,
c$omx+ c)
        a = b + c
c$omx end parallel
      end subroutine

! The comment character is normalized to '!' and the omx sentinel is preserved.
! CHECK-E:{{^}}!$omx parallel
! CHECK-E:{{^}}!$omx barrier
! CHECK-E:{{^}}!$omx end parallel
! CHECK-E:{{^}}!$omx parallel private(a, b, c)
! CHECK-E:{{^}}!$omx end parallel

! With -fopenmp the extension directives are parsed as OpenMP directives.
! CHECK-OMP: !$OMP PARALLEL
! CHECK-OMP: !$OMP BARRIER
! CHECK-OMP: !$OMP END PARALLEL
! CHECK-OMP: !$OMP PARALLEL PRIVATE(a,b,c)
! CHECK-OMP: !$OMP END PARALLEL

! Without -fopenmp the extension directives are ignored as comments.
! CHECK-NO-OMP: SUBROUTINE omx_forms
! CHECK-NO-OMP-NOT: !$OMP
