! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.3 single Construct
! Copyprivate variable is not thread private or private in outer context

program omp_single
  integer i
  i = 10

  !$omp parallel
    !$omp single
    print *, "omp single", i
    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(i)
  !$omp end parallel

end program omp_single
