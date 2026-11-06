! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine selected_index(x)
  implicit none
  integer i, j, x

  !ERROR: Loop index 3 is larger than the generated loops by the INTERCHANGE directive for the INTERCHANGED modifier (2)
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp interchange apply(interchanged(3): tile sizes(2))
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: Loop index 2 is larger than the generated loops by the TILE directive for the GRID modifier (1)
  !BECAUSE: SIZES clause was specified with 1 arguments
  !$omp tile sizes(2) apply(grid(2): reverse)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: Loop index 2 is larger than the generated loops by the TILE directive for the INTRATILE modifier (1)
  !BECAUSE: SIZES clause was specified with 1 arguments
  !$omp tile sizes(2) apply(intratile(2): unroll)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: Loop index 2 is larger than the generated loops by the FUSE directive for the FUSED modifier (1)
  !$omp fuse apply(fused(2): reverse)
  do i = 1, 10
    x = x + 1
  end do
  do j = 1, 10
    x = x - 1
  end do
  !$omp end fuse

end subroutine

