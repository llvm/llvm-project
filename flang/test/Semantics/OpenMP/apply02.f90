! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine wrong_modifier(x)
  implicit none
  integer i, j, x

  !ERROR: Must be a constant value
  !$omp tile sizes(2,2) apply(grid(x): unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: The loop modifier indexes of the APPLY clause must be constant positive integer expressions
  !$omp tile sizes(2,2) apply(grid(-1): unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: The loop modifier indexes of the APPLY clause must be in ascending order
  !$omp tile sizes(2,2) apply(grid(2,1): unroll, nothing)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

end subroutine

