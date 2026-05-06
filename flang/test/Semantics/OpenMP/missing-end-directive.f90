! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! Test that we can diagnose missing end directives without an explosion of errors

! ERROR: Expected OpenMP END PARALLEL directive
!$omp parallel
! ERROR: Expected OpenMP END TASK directive
!$omp task
! ERROR: Expected OpenMP END SECTIONS directive
!$omp sections
! ERROR: Expected OpenMP END PARALLEL directive
!$omp parallel
! ERROR: Expected OpenMP END TASK directive
!$omp task
! ERROR: Expected OpenMP END SECTIONS directive
!$omp sections
end
