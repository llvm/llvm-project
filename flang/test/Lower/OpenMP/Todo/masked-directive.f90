! This test checks lowering of OpenMP masked Directive.

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled directive masked
subroutine test_masked()
  integer :: c = 1
  !$omp masked
  c = c + 1
  !$omp end masked
end subroutine

