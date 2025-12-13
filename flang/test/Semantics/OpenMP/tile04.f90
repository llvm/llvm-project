! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine threads_zero
  implicit none
  integer i

  !ERROR: The parameter of the NUM_THREADS clause must be a positive integer expression
  !$omp parallel do num_threads(-1)
  do i = 1, 5
    print *, i
  end do
end subroutine


subroutine sizes_zero
  implicit none
  integer i

  !ERROR: The parameter of the SIZES clause must be a positive integer expression
  !$omp tile sizes(0)
  do i = 1, 5
    print *, i
  end do
end subroutine


subroutine sizes_negative
  implicit none
  integer i

  !ERROR: The parameter of the SIZES clause must be a positive integer expression
  !$omp tile sizes(-1)
  do i = 1, 5
    print *, i
  end do
end subroutine
