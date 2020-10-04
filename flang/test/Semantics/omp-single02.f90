! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.3 single Construct
! Copyprivate variable is not thread private or private in outer context

program omp_single
  integer i
  i = 10

  !$omp parallel
    !$omp single
    print *, "omp single", i
    !ERROR: copyprivate variable ‘i’ is not threadprivate or private
    !$omp end single copyprivate(i)
  !$omp end parallel

end program omp_single
