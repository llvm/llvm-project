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

  !ERROR: The loop range indicated in the LOOPRANGE(5,2) clause must not be out of the bounds of the Loop Sequence following the construct.
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
end subroutine
