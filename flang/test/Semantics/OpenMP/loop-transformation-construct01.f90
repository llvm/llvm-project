! Testing the Semantics of nested Loop Transformation Constructs

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine loop_transformation_construct1
  implicit none

  !$omp do
  !ERROR: A DO loop must follow the UNROLL directive
  !$omp unroll
end subroutine

subroutine loop_transformation_construct2
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  !$omp tile
  do x = 1, i
    v(x) = x(x) * 2
  end do
  !$omp end tile
  !$omp end do
  !ERROR: The END TILE directive must follow the DO loop associated with the loop construct
  !$omp end tile
end subroutine

subroutine loop_transformation_construct2
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  !ERROR: Only Loop Transformation Constructs or Loop Nests can be nested within Loop Constructs
  !$omp parallel do
  do x = 1, i
    v(x) = x(x) * 2
  end do
end subroutine

subroutine loop_transformation_construct3
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  do x = 1, i
    v(x) = x(x) * 2
  end do
  !ERROR: A DO loop must follow the TILE directive
  !$omp tile
end subroutine

subroutine loop_transformation_construct4
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  !ERROR: If a loop construct has been fully unrolled, it cannot then be tiled
  !$omp tile
  !$omp unroll full
  do x = 1, i
    v(x) = x(x) * 2
  end do
end subroutine

subroutine loop_transformation_construct5
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  !ERROR: If a loop construct has been fully unrolled, it cannot then be tiled
  !$omp tile
  !$omp unroll
  do x = 1, i
    v(x) = x(x) * 2
  end do
end subroutine

subroutine loop_transformation_construct6
  implicit none
  integer :: i = 5
  integer :: y
  integer :: v(i)

  !$omp do
  !$omp tile
  !$omp unroll partial(2)
  do x = 1, i
    v(x) = x(x) * 2
  end do
end subroutine
