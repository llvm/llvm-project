! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

! OpenMP 5.0 and 5.1 omit the 4.5 DEPEND restriction on parts of variables,
! and do not yet exclude substrings from the general list-item definition.
subroutine locators(c, a)
  character(8) :: c, a(2)
  !$omp task depend(in: c(2:3), a(1)(2:3)) affinity(c(2:3))
  !$omp end task
end subroutine
