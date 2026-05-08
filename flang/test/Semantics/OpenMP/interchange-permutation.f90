! Testing the Semantics of interchange
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine double_permutation
  implicit none
  integer i, j

  !ERROR: At most one PERMUTATION clause can appear on the INTERCHANGE directive
  !$omp interchange permutation(2,1) permutation(2,1)
  do i = 1, 5
  do j = 1, 5
    print *, i
  end do
  end do
end subroutine

subroutine zero_parameter
  implicit none
  integer i, j

  !ERROR: The parameter of the PERMUTATION clause must be a constant positive integer expression
  !$omp interchange permutation(0,1)
  do i = 1, 5
  do j = 1, 5
    print *, i
  end do
  end do
end subroutine


subroutine negative_parameter
  implicit none
  integer i, j

  !ERROR: The parameter of the PERMUTATION clause must be a constant positive integer expression
  !$omp interchange permutation(2,-1)
  do i = 1, 5
  do j = 1, 5
    print *, i
  end do
  end do
end subroutine


subroutine constant_parameter
  implicit none
  integer i, j, a

  !ERROR: Must be a constant value
  !$omp interchange permutation(2,a)
  do i = 1, 5
  do j = 1, 5
    print *, i
  end do
  end do
end subroutine

subroutine insufficient_loops
  implicit none
  integer i

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !ERROR: PERMUTATION clause was specified with 2 arguments
  !$omp interchange permutation(2, 1)
  do i = 1, 5
    print *, i
  end do
end subroutine

subroutine minimum_parameters
  implicit none
  integer i, j

  !ERROR: The PERMUTATION clause must have a length of at least two
  !$omp interchange permutation(1)
  do i = 1, 5
    do j = 1, 5
      print *, i
    end do
  end do
end subroutine

subroutine parameter_number
  implicit none
  integer i, j

  !ERROR: Every integer from 1 must appear in the PERMUTATION clause
  !$omp interchange permutation(1,1)
  do i = 1, 5
    do j = 1, 5
      print *, i
    end do
  end do
end subroutine

subroutine parameter_number2
  implicit none
  integer i, j

  !ERROR: Every integer from 1 must appear in the PERMUTATION clause
  !$omp interchange permutation(1,3)
  do i = 1, 5
    do j = 1, 5
      print *, i
    end do
  end do
end subroutine

