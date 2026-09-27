! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine insufficient_loops
  implicit none
  integer i

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp tile sizes(2, 2)
  do i = 1, 5
    print *, i
  end do
end subroutine
