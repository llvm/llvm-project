! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME:  @[[RED_I32_NAME:.*]] : i32 init {
!CHECK:       ^bb0(%{{.*}}: i32):
!CHECK:         %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:         omp.yield(%[[C0_1]] : i32)
!CHECK:       } combiner {
!CHECK:      ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:        %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:        omp.yield(%[[RES]] : i32)
!CHECK:       }

!CHECK-LABEL: func.func @_QPomp_taskgroup_task_reduction() {
!CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskgroup_task_reductionEres"}
!CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFomp_taskgroup_task_reductionEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.taskgroup task_reduction(@[[RED_I32_NAME]]  %[[VAL_1]]#0 -> %[[VAL_2:.*]] : !fir.ref<i32>) {
!CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] 
!CHECK-SAME:      {uniq_name = "_QFomp_taskgroup_task_reductionEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_4]], %[[VAL_5]] : i32
!CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
!CHECK:           omp.terminator
!CHECK:          }
!CHECK:        return
!CHECK:       }


subroutine omp_taskgroup_task_reduction()
   integer :: res
   !$omp taskgroup task_reduction(+:res)
   res = res + 1
   !$omp end taskgroup
end subroutine
