! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! A `target in_reduction` list item that is a COMMON block member is accessed in
! the target body through the COMMON block storage map, not through its own
! dedicated member map entry. The host-side redirect only rewrites the
! member map argument, so the runtime-private pointer would be dead and the body
! would accumulate into the mapped original instead of the reduction-private
! copy. Until the redirect can target the storage map this case is rejected.

! CHECK: not yet implemented: TARGET construct with IN_REDUCTION of a COMMON block member

subroutine omp_target_in_reduction_common()
  integer :: i, k
  common /cb/ i, k
  i = 0
  !$omp target map(tofrom: /cb/) in_reduction(+:i)
  i = i + 1
  !$omp end target
end subroutine omp_target_in_reduction_common
