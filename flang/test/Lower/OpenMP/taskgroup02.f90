! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Check that variables are not privatized twice when TASKGROUP is used.

!CHECK-LABEL: func.func @_QPsub() {
!CHECK:         omp.parallel {
!CHECK:           %[[PAR_I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFsubEi"}
!CHECK:           omp.master {
!CHECK:             omp.taskgroup {
!CHECK-NEXT:          omp.task private(@_QFsubEi_firstprivate_i32 %[[PAR_I]]#0 -> %[[TASK_I:.*]] : !fir.ref<i32>) {
!CHECK:                 %[[TASK_I_DECL:.*]]:2 = hlfir.declare %[[TASK_I]] {uniq_name = "_QFsubEi"}
!CHECK:               }
!CHECK:             }
!CHECK:           }
!CHECK:         }

subroutine sub()
  integer, dimension(10) :: a
  integer :: i

  !$omp parallel
    !$omp master
      do i=1,10
       !$omp taskgroup
         !$omp task shared(a)
           a(i) = 1
         !$omp end task
       !$omp end taskgroup
      end do
    !$omp end master
  !$omp end parallel
end subroutine
