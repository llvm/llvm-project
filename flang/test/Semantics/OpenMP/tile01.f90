! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine missing_sizes
  implicit none
  integer i

  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
  do i = 1, 42
    print *, i
  end do
end subroutine


subroutine double_sizes
  implicit none
  integer i

  !ERROR: At most one SIZES clause can appear on the TILE directive
  !$omp tile sizes(2) sizes(2)
  do i = 1, 5
    print *, i
  end do
end subroutine
