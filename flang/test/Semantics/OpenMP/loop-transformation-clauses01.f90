! Testing the Semantics of clauses on loop transformation directives

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60


subroutine loop_transformation_construct1
  implicit none
  integer, parameter:: i = 5
  integer :: x
  integer :: a
  integer :: v(i)

  !ERROR: At most one LOOPRANGE clause can appear on the FUSE directive
  !$omp fuse looprange(1,2) looprange(1,2)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  !ERROR: The specified loop range requires 6 loops, but the loop sequence has a length of 2
  !$omp fuse looprange(5,2)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  !ERROR: The parameter of the LOOPRANGE clause must be a constant positive integer expression
  !$omp fuse looprange(0,1)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  !ERROR: The parameter of the LOOPRANGE clause must be a constant positive integer expression
  !$omp fuse looprange(1,-1)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  !ERROR: Must be a constant value
  !$omp fuse looprange(a,2)
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  !ERROR: Must be a constant value
  !$omp fuse looprange(1,a)
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse

  ! This is ok aside from the warnings about compiler directives
  !$omp fuse looprange(1,3)
    do x = 1, 10; end do        ! 1 loop
    !WARNING: Compiler directives are not allowed inside OpenMP loop constructs
    !dir$ novector
    !$omp fuse looprange(1,2)   ! 2 loops
      do x = 1, 10; end do
      !WARNING: Compiler directives are not allowed inside OpenMP loop constructs
      !dir$ nounroll
      do x = 1, 10; end do
      do x = 1, 10; end do
    !$omp end fuse
  !$omp end fuse
end subroutine
