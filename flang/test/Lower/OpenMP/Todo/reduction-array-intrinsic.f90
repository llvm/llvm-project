! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Reduction of some types is not supported for intrinsics
subroutine max_array_reduction(l, r)
  integer :: l(:), r(:)

  !$omp parallel reduction(max:l)
    l = max(l, r)
  !$omp end parallel
end subroutine
