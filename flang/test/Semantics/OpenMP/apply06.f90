! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine full_unroll(x)
  implicit none
  integer i, j, x, k

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp interchange
  do i = 1, 10
    !BECAUSE: Fully unrolled loop does not result in a loop nest
    !$omp interchange apply(interchanged(1): unroll)
    do j = 1, 10
      do k = 1, 10
        x = x + 1
      end do
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest at index 1 is a perfect nest of depth 1
  !BECAUSE: Fully unrolled loop does not result in a loop nest
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp interchange permutation(2,1) apply(interchanged: tile sizes(2,2), unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: This construct requires a canonical loop nest
  !$omp interchange
  !BECAUSE: Fully unrolled loop does not result in a loop nest
  !$omp interchange apply(interchanged(1): unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

end subroutine

