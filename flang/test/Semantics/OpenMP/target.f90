! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Werror

! Corner cases in OpenMP target directives

subroutine empty_target()
  integer :: i

  !$omp target map(i)
  !$omp end target

end subroutine
