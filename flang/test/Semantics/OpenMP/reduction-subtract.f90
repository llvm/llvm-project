! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! Minus operation is deprecated in reduction

subroutine reduction_subtract
  integer :: x
  !ERROR: Invalid reduction operator in REDUCTION clause
  !ERROR: The minus reduction operator is deprecated since OpenMP 5.2
  !$omp do reduction(-:x)
  do i=1, 100
    x = x - i
  end do
  !$omp end do
end subroutine
