! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause COLLAPSE in TASKLOOP construct
subroutine omp_taskloop_collapse()
   integer x
   x = 0
   !$omp taskloop collapse(2)
   do i = 1, 100
     do j = 1, 100
      x = x + 1
     end do
   end do
   !$omp end taskloop
end subroutine omp_taskloop_collapse
