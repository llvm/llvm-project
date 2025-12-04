!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

subroutine f(x)
  integer :: x
!ERROR: PARALLEL cannot follow SECTIONS
!$omp sections parallel
!$omp section
  x = x + 1
!$omp end sections
end
end
