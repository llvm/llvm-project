! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

! The "allocate" clause has been removed, because it needs to be used
! together with a privatizing clause. The only such clause for "taskgroup"
! is "task_reduction", but it's not yet supported.

!CHECK-LABEL: @_QPomp_taskgroup
subroutine omp_taskgroup
!CHECK: omp.taskgroup
!$omp taskgroup
!CHECK: omp.task
!$omp task
!CHECK: fir.call @_QPwork() {{.*}}: () -> ()
   call work()
!CHECK: omp.terminator
!$omp end task
!CHECK: omp.terminator
!$omp end taskgroup
end subroutine
