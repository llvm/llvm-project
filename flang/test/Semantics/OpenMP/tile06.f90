! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine nonrectangular_loop_lb
  implicit none
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp tile sizes(2,2)
  do i = 1, 5
    do j = 1, i
      print *, i, j
    end do
  end do
end subroutine


subroutine nonrectangular_loop_ub
  implicit none
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp tile sizes(2,2)
  do i = 1, 5
    do j = 1, i
      print *, i, j
    end do
  end do
end subroutine


subroutine nonrectangular_loop_step
  implicit none
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp tile sizes(2,2)
  do i = 1, 5
    do j = 1, 42, i
      print *, i, j
    end do
  end do
end subroutine
