! RUN: bbc -emit-hlfir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s -fopenmp-version=50 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private {type = private} 
! CHECK-SAME:        @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL: func.func @_QPomp_taskloop() {
! CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloopEi"}
! CHECK:         %[[DECL_I:.*]]:2 = hlfir.declare %1 {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         omp.parallel {
! CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:           %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:           %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:           omp.taskloop private(@[[I_PRIVATE]] %2#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
! CHECK:             omp.loop_nest (%[[ARG1:.*]]) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:               %[[IDX:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               hlfir.assign %[[ARG1]] to %[[IDX]]#0 : i32, !fir.ref<i32>
! CHECK:               omp.cancel cancellation_construct_type(taskgroup)
! CHECK:               omp.yield
! CHECK:             }
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine omp_taskloop
integer :: i
!$omp parallel
    !$omp taskloop
      do i = 1, 10
      !$omp cancel taskgroup
      end do
    !$omp end taskloop
!$omp end parallel
end subroutine omp_taskloop
