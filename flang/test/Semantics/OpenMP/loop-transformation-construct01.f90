! Testing the Semantics of nested Loop Transformation Constructs

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine loop_transformation_construct1
  implicit none

  !ERROR: OpenMP loop construct cannot apply to a fully unrolled loop
  !$omp do
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp unroll
end subroutine

subroutine loop_transformation_construct2
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !$omp end tile
  !$omp end do
end subroutine

subroutine loop_transformation_construct3
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: Only loop-transforming OpenMP constructs are allowed inside OpenMP loop constructs
  !$omp parallel do
  do x = 1, i
    v(x) = v(x) * 2
  end do
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
  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
end subroutine

subroutine loop_transformation_construct5
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: OpenMP loop construct cannot apply to a fully unrolled loop
  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
  !$omp unroll full
  do x = 1, i
    v(x) = v(x) * 2
  end do
end subroutine

subroutine loop_transformation_construct6
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: OpenMP loop construct cannot apply to a fully unrolled loop
  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
  !$omp unroll
  do x = 1, i
    v(x) = v(x) * 2
  end do
end subroutine

subroutine loop_transformation_construct7
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: At least one of SIZES clause must appear on the TILE directive
  !$omp tile
  !$omp unroll partial(2)
  do x = 1, i
    v(x) = v(x) * 2
  end do
end subroutine
