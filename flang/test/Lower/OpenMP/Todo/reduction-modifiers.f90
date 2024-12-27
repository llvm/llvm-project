! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction modifiers are not supported

subroutine foo()
  integer :: i, j
  j = 0
  !$omp do reduction (inscan, *: j)
  do i = 1, 10
    !$omp scan inclusive(j)
    j = j + 1
  end do
end subroutine
