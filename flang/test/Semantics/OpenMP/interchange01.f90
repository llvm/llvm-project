! Testing the Semantics of interchange
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine on_unroll
  implicit none
  integer i, j

  !ERROR: OpenMP loop construct cannot apply to a fully unrolled loop
  !$omp interchange
  !$omp unroll
  do i = 1, 5
    do j = 1, 5
      print *, i
    end do
  end do
end subroutine

subroutine loop_assoc
  implicit none
  integer :: i, j

  !$omp interchange
  !ERROR: The associated loop of a loop-associated directive cannot be a DO WHILE.
  do while (i <= 10)
    do j = 1, 5
      i = i + 1
      print *, i
    end do
  end do
end subroutine

subroutine insufficient_loops
  implicit none
  integer i

  !ERROR: The INTERCHANGE construct must be followed by a canonical loop nest of at least 2 levels
  !$omp interchange 
  do i = 1, 5
    print *, i
  end do
end subroutine

