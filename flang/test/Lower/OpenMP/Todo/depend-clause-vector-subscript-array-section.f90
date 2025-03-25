! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: Vector subscripted array section for task dependency
subroutine vectorSubscriptArraySection(array, indices)
  integer :: array(:)
  integer :: indices(:)

  !$omp task depend (in: array(indices))
  !$omp end task
end subroutine
