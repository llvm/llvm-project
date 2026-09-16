! Test lowering of OpenMP metadirective with static user={condition()}
! selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | \
! RUN:   FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | \
! RUN:   FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | \
! RUN:   FileCheck %s

! Large non-negative scores must not wrap their sum to zero.
! CHECK-LABEL: func.func @_QPtest_wide_score()
! CHECK: omp.parallel
! CHECK-NOT: omp.taskyield
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine test_wide_score()
  !$omp parallel
    !$omp metadirective &
    !$omp& when(user={condition(score(9223372036854775807_8): .true.)}, &
    !$omp& implementation={vendor(score(9223372036854775807_8): llvm)}, &
    !$omp& construct={parallel}: barrier) &
    !$omp& when(user={condition(.true.)}: taskyield)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_true()
! CHECK:         omp.taskyield
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_true()
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: taskyield)
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_false()
! CHECK-NOT:     omp.taskwait
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_false()
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_score()
! CHECK-NOT:     omp.taskyield
! CHECK:         omp.taskwait
! CHECK:         return
subroutine test_condition_score()
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: taskyield) &
  !$omp & when(user={condition(score(2): .true.)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_condition_true()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_begin_condition_true()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(user={condition(.true.)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_condition_false()
! CHECK-NOT:     omp.parallel
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_begin_condition_false()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(user={condition(.false.)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine
