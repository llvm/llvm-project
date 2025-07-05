! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME:  @[[RED_I32_NAME:.*]] : i32 init {
!CHECK:       ^bb0(%{{.*}}: i32):
!CHECK:         %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:         omp.yield(%[[C0_1]] : i32)
!CHECK:       } combiner {
!CHECK:       ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:          %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:          omp.yield(%[[RES]] : i32)
!CHECK:         }

!CHECK-LABEL:  func.func @_QPomp_task_in_reduction() {
!                [...]
!CHECK:          omp.task in_reduction(@[[RED_I32_NAME]] %[[VAL_1:.*]]#0  -> %[[ARG0]] : !fir.ref<i32>) {
!CHECK:            %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]]
!CHECK-SAME:       {uniq_name = "_QFomp_task_in_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:            %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
!CHECK:            %[[VAL_6:.*]] = arith.constant 1 : i32
!CHECK:            %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
!CHECK:            hlfir.assign %[[VAL_7]] to %[[VAL_4]]#0 : i32, !fir.ref<i32>
!CHECK:            omp.terminator
!CHECK:           }
!CHECK:           return
!CHECK:          }

subroutine omp_task_in_reduction()
   integer i
   i = 0
   !$omp task in_reduction(+:i)
   i = i + 1
   !$omp end task
end subroutine omp_task_in_reduction
