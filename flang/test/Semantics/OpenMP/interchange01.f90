! Testing the Semantics of interchange
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine on_unroll
  implicit none
  integer i, j

  !ERROR: This construct requires a canonical loop nest
  !$omp interchange
  !BECAUSE: Fully unrolled loop does not result in a loop nest
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

  !ERROR: This construct requires a canonical loop nest
  !$omp interchange
  !BECAUSE: DO WHILE loop is not a valid affected loop
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

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: PERMUTATION clause was not specified, a permutation (2, 1) is assumed
  !$omp interchange 
  do i = 1, 5
    print *, i
  end do
end subroutine

