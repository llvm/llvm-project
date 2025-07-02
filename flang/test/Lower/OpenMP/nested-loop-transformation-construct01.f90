! Test to ensure TODO message is emitted for tile OpenMP 5.1 Directives when they are nested.

!RUN: not %flang -fopenmp -fopenmp-version=51 %s 2>&1 | FileCheck %s

subroutine loop_transformation_construct
  implicit none
  integer :: I = 10
  integer :: x
  integer :: y(I)

  !$omp do
  !$omp tile
  do i = 1, I
    y(i) = y(i) * 5
  end do
  !$omp end tile
  !$omp end do
end subroutine

!CHECK: not yet implemented: Unhandled loop directive (tile)
