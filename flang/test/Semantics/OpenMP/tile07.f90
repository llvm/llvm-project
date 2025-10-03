! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine non_perfectly_nested_loop_behind
  implicit none
  integer i, j

  !ERROR: Canonical loop nest must be perfectly nested.
  !$omp tile sizes(2,2)
  do i = 1, 5
    do j = 1, 42
      print *, j
    end do
    print *, i
  end do
end subroutine


subroutine non_perfectly_nested_loop_before
  implicit none
  integer i, j

  !ERROR: The SIZES clause has more entries than there are nested canonical loops.
  !$omp tile sizes(2,2)
  do i = 1, 5
    print *, i
    do j = 1, 42
      print *, j
    end do
  end do
end subroutine



