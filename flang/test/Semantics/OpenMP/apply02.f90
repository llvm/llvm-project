! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine bad_index(x)
  implicit none
  integer i, j, x

  !ERROR: Must be a constant value
  !ERROR: The number of selected loops (0) and the number of directives (1) must match
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

  !ERROR: Loop index 1 is selected multiple times for the GRID modifier
  !ERROR: This construct requires a perfect nest of depth 1, but the associated nest at index 1 is a perfect nest of depth 0
  !BECAUSE: Fully unrolled loop does not result in a loop nest
  !$omp tile sizes(2,2) apply(grid(1): unroll) apply(grid(1): reverse)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: Loop index 1 is selected multiple times for the GRID modifier
  !BECAUSE: All generated loops (2) are selected because no indexes were specified
  !ERROR: This construct requires a perfect nest of depth 1, but the associated nest at index 1 is a perfect nest of depth 0
  !BECAUSE: Fully unrolled loop does not result in a loop nest
  !$omp tile sizes(2,2) apply(grid(1): unroll) apply(grid: reverse, unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

end subroutine

