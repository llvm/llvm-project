! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine on_unroll
  implicit none
  integer i

  !ERROR: If a loop construct has been fully unrolled, it cannot then be tiled
  !$omp tile sizes(2)
  !$omp unroll
  do i = 1, 5
    print *, i
  end do
end subroutine
