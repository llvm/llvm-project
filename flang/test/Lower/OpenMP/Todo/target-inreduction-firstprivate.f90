! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! An explicit `firstprivate(i)` on the same `target` as `in_reduction(+:i)`
! privatizes the reduction variable. The item is then privatized instead of
! address-preserving-mapped, so the host-side redirect has no dedicated map
! argument to rewrite and the firstprivate copy region has no mold value, which
! previously aborted translation. Reject until in_reduction can be reconciled
! with target privatization.

! CHECK: not yet implemented: TARGET construct with IN_REDUCTION of a privatized variable

subroutine omp_target_in_reduction_firstprivate()
  integer :: i
  i = 0
  !$omp target firstprivate(i) in_reduction(+:i)
  i = i + 1
  !$omp end target
end subroutine omp_target_in_reduction_firstprivate
