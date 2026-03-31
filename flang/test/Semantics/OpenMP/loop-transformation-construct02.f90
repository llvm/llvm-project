! Testing the Semantics of loop sequences combined with 
! nested Loop Transformation Constructs

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine loop_transformation_construct1
  implicit none

  !$omp do
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp fuse 
end subroutine

subroutine loop_transformation_construct2
  implicit none

  !$omp do
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp fuse 
  !$omp end fuse
end subroutine

subroutine loop_transformation_construct3
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !$omp fuse
  do x = 1, i
    v(x) = v(x) * 2
  end do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !$omp end fuse
  !$omp end do
  !ERROR: Misplaced OpenMP end-directive
  !$omp end fuse
end subroutine

subroutine loop_transformation_construct4
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp fuse
  !$omp end fuse
end subroutine

subroutine loop_transformation_construct5
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: OpenMP loop construct cannot apply to a fully unrolled loop
  !$omp fuse
  !$omp unroll full
  do x = 1, i
    v(x) = v(x) * 2
  end do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !$omp end fuse
end subroutine

subroutine loop_transformation_construct6
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !ERROR: This construct applies to a loop nest, but has a loop sequence of length 2
  !$omp do
  !$omp fuse looprange(1,1)
  !$omp unroll partial(2)
  do x = 1, i
    v(x) = v(x) * 2
  end do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !$omp end fuse 
end subroutine
