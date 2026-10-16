! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! a common block in a data-sharing clause is equivalent to
! listing every explicit member of the common block.

subroutine common_block_dsa()
  common /c/ x, y
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel private(/c/) shared(x)
  !$omp end parallel
end subroutine
