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
  !ERROR: FUSED modifier is not allowed on TILE directive
  !$omp tile sizes(2,2) apply(interchanged: nothing, reverse) apply(fused: reverse)
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
end subroutine

