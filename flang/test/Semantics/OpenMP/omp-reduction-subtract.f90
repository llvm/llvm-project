! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! Minus operation is deprecated in reduction

subroutine reduction_subtract
  integer :: x
  !ERROR: The minus reduction operator is deprecated since OpenMP 5.2 and is not supported in the REDUCTION clause.
  !$omp do reduction(-:x)
  do i=1, 100
    x = x - i
  end do
  !$omp end do
end subroutine
