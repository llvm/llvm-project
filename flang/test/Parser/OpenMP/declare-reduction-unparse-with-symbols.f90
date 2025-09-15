!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s | FileCheck %s

! This used to crash.

subroutine f00
  !$omp declare reduction(fred : integer, real : omp_out = omp_in + omp_out)
end

!CHECK: !DEF: /f00 (Subroutine) Subprogram
!CHECK: subroutine f00
!CHECK: !$omp declare reduction (fred:integer,real:omp_out = omp_in+omp_out)
!CHECK: end subroutine

