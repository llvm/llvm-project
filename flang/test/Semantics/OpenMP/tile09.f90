! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00
  integer :: i, j
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp tile sizes(2, 2)
  do i = 1, 10
    !BECAUSE: This construct is not a DO-loop or a loop-nest-generating construct
    !$omp do
    do j = 1, 10
    end do
  end do
end

subroutine f01
  integer :: i, j
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp tile sizes(2, 2)
  do i = 1, 10
    !BECAUSE: Fully unrolled loop does not result in a loop nest
    !$omp unroll full
    do j = 1, 10
    end do
  end do
end

subroutine f02
  integer :: i, j
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp tile sizes(2, 2)
  !BECAUSE: Partially unrolled loop cannot form a nest of depth > 1
  !$omp unroll partial
  do i = 1, 10
    do j = 1, 10
    end do
  end do
end

subroutine f03
  integer :: i, j, k
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 3 arguments
  !$omp tile sizes(2, 2, 2)
  !BECAUSE: This construct does not contain a loop nest
  do i = 1, 10
    !BECAUSE: LOOPRANGE clause was specified with a count of 1 starting at loop 1
    !BECAUSE: This FUSE construct does not result in a loop nest, but a proper loop sequence
    !$omp fuse depth(2) looprange(1, 1)
    do j = 1, 10
      do k = 1, 10
      end do
    end do
    do j = 1, 10
      do k = 1, 10
      end do
    end do
    !$omp end fuse
  end do
end

