! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unsupported atomic type
subroutine complex_atomic
  complex :: l, r
  !$omp atomic read
    l = r
end subroutine
