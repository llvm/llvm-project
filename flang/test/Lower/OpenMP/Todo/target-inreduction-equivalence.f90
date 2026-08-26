! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! A `target in_reduction` list item that is EQUIVALENCE storage-associated
! shares storage with another variable that may be mapped separately (here `j`).
! The body updates the aliased storage through `j`'s map while the host-side
! redirect rewrites only `i`'s map argument, leaving the runtime-private copy
! dead. Reject until the redirect can account for storage association.

! CHECK: not yet implemented: TARGET construct with IN_REDUCTION of an EQUIVALENCE storage-associated variable

subroutine omp_target_in_reduction_equivalence()
  integer :: i, j
  equivalence(i, j)
  i = 0
  !$omp target map(tofrom: j) in_reduction(+:i)
  j = j + 1
  !$omp end target
end subroutine omp_target_in_reduction_equivalence
