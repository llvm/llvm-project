! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine non_perfectly_nested_loop_behind
  implicit none
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
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

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: SIZES clause was specified with 2 arguments
  !$omp tile sizes(2,2)
  do i = 1, 5
    print *, i
    do j = 1, 42
      print *, j
    end do
  end do
end subroutine



