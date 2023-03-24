! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Common Block in privatization clause
subroutine lastprivate_common
  common /c/ x, y
  real x, y
  !$omp do lastprivate(/c/)
  do i=1,100
  end do
  !$omp end do
end subroutine
