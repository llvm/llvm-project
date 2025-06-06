! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause REDUCTION in TASKLOOP construct
subroutine omp_taskloop_reduction()
   integer x
   x = 0
   !$omp taskloop reduction(+:x)
   do i = 1, 100
      x = x + 1
   end do
   !$omp end taskloop
end subroutine omp_taskloop_reduction
