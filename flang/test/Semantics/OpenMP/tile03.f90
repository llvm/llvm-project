! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine loop_assoc
  implicit none
  integer :: i = 0

  !$omp tile sizes(2)
  !ERROR: The associated loop of a loop-associated directive cannot be a DO WHILE.
  do while (i <= 10)
    i = i + 1
    print *, i
  end do
end subroutine
