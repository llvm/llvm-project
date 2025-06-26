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
  !ERROR: Only OpenMP Loop Transformation Constructs can be nested within OpenMPLoopConstruct's
  !$omp parallel do
  do x = 1, i
    v(x) = x(x) * 2
  end do
  !! This error occurs because the `parallel do` end directive never gets matched.
  !ERROR: The END PARALLEL DO directive must follow the DO loop associated with the loop construct
  !$omp end parallel do
  !$omp end do
end subroutine
