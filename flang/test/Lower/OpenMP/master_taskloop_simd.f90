! This test checks lowering of OpenMP master taskloop simd Directive.

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine test_master_taskloop_simd()
  integer :: i, j = 1
  !CHECK: not yet implemented: Composite TASKLOOP SIMD
  !$omp master taskloop simd 
  do i=1,10
   j = j + 1
  end do
  !$omp end master taskloop simd
end subroutine
