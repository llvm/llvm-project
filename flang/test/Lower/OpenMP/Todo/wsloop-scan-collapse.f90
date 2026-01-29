! Tests scan reduction behavior when used in nested workshare loops

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program nested_loop_example
  implicit none
  integer :: i, j, x
  integer, parameter :: N = 100, M = 200
  real :: A(N, M), B(N, M)
  x = 0

  do i = 1, N
    do j = 1, M
      A(i, j) = i * j
    end do
  end do
  
  !$omp parallel do collapse(2) reduction(inscan, +:x)
  do i = 1, N
    do j = 1, M
      x = x + A(i,j)
      !CHECK: not yet implemented: Scan directive inside nested workshare loops
      !$omp scan inclusive(x)
      B(i, j) = x
    end do
  end do
  !$omp end parallel do

end program nested_loop_example
