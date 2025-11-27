! Testing the Semantic failure of forming loop sequences under regular OpenMP directives 

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine loop_transformation_construct3
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !ERROR: The loop sequence following the DO construct must be fully fused first.
  !$omp do
  !$omp fuse looprange(1,2)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse
  !$omp end do
end subroutine

subroutine loop_transformation_construct4
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !ERROR: The loop sequence following the TILE construct must be fully fused first.
  !$omp tile sizes(2)
  !$omp fuse looprange(1,2)
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  do x = 1, i
    v(x) = x * 2
  end do
  !$omp end fuse
  !$omp end tile
end subroutine
