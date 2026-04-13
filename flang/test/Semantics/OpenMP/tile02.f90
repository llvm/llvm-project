! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine on_unroll
  implicit none
  integer i

  !ERROR: This construct requires a canonical loop nest
  !$omp tile sizes(2)
  !BECAUSE: Fully unrolled loop does not result in a loop nest
  !$omp unroll
  do i = 1, 5
    print *, i
  end do
end subroutine
