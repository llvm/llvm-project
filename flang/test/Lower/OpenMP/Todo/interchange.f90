! Tests reduction processor behavior when a reduction symbol is not supported.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine foo
  implicit none
  integer :: j, i

  !CHECK: not yet implemented: OpenMP Interchange
  !$omp interchange
  do i=1,10
    do j=1,10
    end do
  end do
end subroutine
