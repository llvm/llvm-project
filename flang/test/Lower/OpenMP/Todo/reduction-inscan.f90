! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction modifiers are not supported
subroutine reduction_inscan()
  integer :: i,j
  i = 0

  !$omp do reduction(inscan, +:i)
  do j=1,10
     !$omp scan inclusive(i)
     i = i + 1
  end do
  !$omp end do
end subroutine reduction_inscan
