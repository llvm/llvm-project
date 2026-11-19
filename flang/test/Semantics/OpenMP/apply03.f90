! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine bad_clause(x)
  implicit none
  integer i, j, x

  !ERROR: The number of selected loops (2) and the number of directives (1) must match
  !$omp interchange apply(interchanged(1,2): tile sizes(2))
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: The number of selected loops (2) and the number of directives (1) must match
  !BECAUSE: All generated loops (2) are selected because no indexes were specified
  !$omp interchange apply(interchanged: tile sizes(2))
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

  !ERROR: Only Loop Transformation directives are allowed in the APPLY clause directive list
  !$omp interchange apply(interchanged(1): do)
  do i = 1, 10
    do j = 1, 10
      x = x + 1
    end do
  end do

end subroutine

