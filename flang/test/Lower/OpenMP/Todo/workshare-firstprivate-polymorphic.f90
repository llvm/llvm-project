! Tests that firstprivate of a polymorphic variable in a parallel workshare
! region is not yet supported.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

module shapes
  type :: shape
    real :: area
  end type
end module

subroutine poly_workshare_firstprivate(p, a, n)
  use shapes
  implicit none
  integer :: n
  class(shape), allocatable :: p(:)
  real :: a(n)

  !CHECK: not yet implemented: create polymorphic host associated copy
  !$omp parallel workshare firstprivate(p)
    a = p%area + 1.0
  !$omp end parallel workshare
end subroutine
