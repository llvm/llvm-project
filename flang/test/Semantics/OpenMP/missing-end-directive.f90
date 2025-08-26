! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! Test that we can diagnose missing end directives without an explosion of errors

! ERROR: Expected OpenMP end directive
!$omp parallel
! ERROR: Expected OpenMP end directive
!$omp task
! ERROR: Expected OpenMP END SECTIONS directive
!$omp sections
! ERROR: Expected OpenMP end directive
!$omp parallel
! ERROR: Expected OpenMP end directive
!$omp task
! ERROR: Expected OpenMP END SECTIONS directive
!$omp sections
end
