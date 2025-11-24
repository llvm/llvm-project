! Testing the Semantics of loop sequences combined with 
! nested Loop Transformation Constructs

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60
!XFAIL: *

subroutine loop_transformation_construct1
  implicit none

  !$omp do
  !ERROR: The FUSE construct requires the END FUSE directive
  !$omp fuse 
end subroutine

subroutine loop_transformation_construct2
  implicit none

  !$omp do
  !ERROR: A DO loop must follow the FUSE directive
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
  !ERROR: The END FUSE directive must follow the DO loop associated with the loop construct
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
  !ERROR: A DO loop must follow the FUSE directive
  !$omp fuse
  !$omp end fuse
end subroutine

subroutine loop_transformation_construct5
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  !$omp do
  !ERROR: If a loop construct has been fully unrolled, it cannot then be further transformed
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
