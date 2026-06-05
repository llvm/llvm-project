! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine depth(x)
  implicit none
  integer i, j, x, k

  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest at index 1 is a perfect nest of depth 2
  !BECAUSE: SIZES clause was specified with 3 arguments
  !$omp interchange apply(interchanged(1): tile sizes(2,2,2))
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest at index 1 is a perfect nest of depth 1
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp tile sizes(2) apply(intratile: interchange)
  do i = 1, 10
    x = x + 1
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest at index 3 is a perfect nest of depth 1
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp interchange permutation(3,2,1) apply(interchanged(3): interchange)
  do i = 1, 10
    !$omp tile sizes(2)
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 4, but the associated nest is a perfect nest of depth 3
  !BECAUSE: PERMUTATION clause was specified with 4 arguments
  !$omp interchange permutation(4,3,2,1)
  do i = 1, 10
    !$omp interchange apply(interchanged(1): interchange)
    do j = 1, 10
      do k = 1, 10
        x = x + 1
      end do
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 5, but the associated nest is a perfect nest of depth 4
  !BECAUSE: PERMUTATION clause was specified with 5 arguments
  !$omp interchange permutation(5,4,3,2,1)
  do i = 1, 10
    !$omp interchange apply(interchanged(1): tile sizes(2))
    do j = 1, 10
      do k = 1, 10
        x = x + 1
      end do
    end do
  end do

  !ERROR: This construct requires a perfect nest of depth 4, but the associated nest at index 1 is a perfect nest of depth 3
  !BECAUSE: PERMUTATION clause was specified with 4 arguments
  !$omp tile sizes(2) apply(intratile: tile sizes(2)) apply(grid: interchange permutation(4,3,2,1))
  do i = 1, 10
    x = x + 1
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest at index 1 is a perfect nest of depth 1
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp fuse apply(fused: interchange)
  do i = 1, 10
    x = x + 1
  end do
  do j = 1, 10
    x = x - 1
  end do
  !$omp end fuse

end subroutine

