! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine loop_assoc
  implicit none
  integer :: i = 0

  !ERROR: This construct requires a canonical loop nest
  !$omp tile sizes(2)
  !BECAUSE: DO WHILE loop is not a valid affected loop
  do while (i <= 10)
    i = i + 1
    print *, i
  end do
end subroutine
