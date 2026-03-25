!RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s
!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: Unhandled clause DEPTH in FUSE construct
subroutine f00
  integer :: i, j
  !$omp fuse depth(2)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  do i = 1, 10
    do j = 1, 10
    end do
  end do
  !$omp end fuse
end

