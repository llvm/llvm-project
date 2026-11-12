! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine wrong_modifier(x)
  implicit none
  integer i, j, x

  !ERROR: INTERCHANGED modifier is not allowed on TILE directive
  !$omp tile sizes(2,2) apply(interchanged: nothing, reverse)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: INTERCHANGED modifier is not allowed on TILE directive
  !ERROR: Loop index 1 is selected multiple times for the FUSED modifier
  !BECAUSE: All generated loops (2) are selected because no indexes were specified
  !ERROR: Loop index 2 is selected multiple times for the FUSED modifier
  !ERROR: FUSED modifier is not allowed on TILE directive
  !$omp tile sizes(2,2) apply(interchanged: nothing, reverse) apply(fused: reverse, nothing)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: INTRATILE modifier is not allowed on UNROLL directive
  !$omp tile sizes(2,2) apply(grid: nothing, unroll partial(2) apply(intratile: reverse))
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: TILE directive does not have a default loop modifier
  !$omp tile sizes(2,2) apply(nothing, unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do
end subroutine

