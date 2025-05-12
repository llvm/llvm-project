!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

subroutine f(x)
  integer :: x
!ERROR: Cancellation construct type is not allowed on SECTIONS
!$omp sections parallel
!$omp section
  x = x + 1
!$omp end sections
end
end
