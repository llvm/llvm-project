! REQUIRES: openmp_runtime
! RUN: %flang_fc1 -emit-fir %openmp_flags -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s

!===============================================================================
! `detach` clause
!===============================================================================

!CHECK: omp.task detach(%[[EVENT_HANDLE:.*]] : !fir.ref<i64>) {
!CHECK: fir.call @_QPfoo() fastmath<contract> : () -> ()
!CHECK: omp.terminator
!CHECK: }
subroutine omp_task_detach()
  use omp_lib
  integer (kind=omp_event_handle_kind) :: event
  !$omp task detach(event)
  call foo()
  !$omp end task
end subroutine omp_task_detach
