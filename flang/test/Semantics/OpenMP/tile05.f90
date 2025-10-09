! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine insufficient_loops
  implicit none
  integer i

  !ERROR: The SIZES clause has more entries than there are nested canonical loops.
  !$omp tile sizes(2, 2)
  do i = 1, 5
    print *, i
  end do
end subroutine
