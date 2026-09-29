! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: omp.declare_reduction
!CHECK-SAME:  @[[RED_I32_NAME:.*]] : i32 init {
!CHECK:       ^bb0(%{{.*}}: i32):
!CHECK:         %[[C0_1:.*]] = arith.constant 0 : i32
!CHECK:         omp.yield(%[[C0_1]] : i32)
!CHECK:       } combiner {
!CHECK:       ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
!CHECK:         %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
!CHECK:         omp.yield(%[[RES]] : i32)
!CHECK:        }

!CHECK-LABEL:  func.func @_QPin_reduction() {
!                [...]
!CHECK:          omp.taskgroup task_reduction(@[[RED_I32_NAME]] %[[VAL_1:.*]]#0 -> %[[VAL_3:.*]] : !fir.ref<i32>) {
!CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFin_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             omp.task in_reduction(@[[RED_I32_NAME]] %[[VAL_4]]#0 -> %[[VAL_5:.*]] : !fir.ref<i32>) {
!CHECK:               %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFin_reductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!                     [...]
!CHECK:               omp.terminator
!CHECK:             }
!CHECK:             omp.terminator
!CHECK:          }
!CHECK:          return
!CHECK:         }

subroutine in_reduction()
   integer :: x
   x = 0
   !$omp taskgroup task_reduction(+:x)
   !$omp task in_reduction(+:x)
   x = x + 1
   !$omp end task
   !$omp end taskgroup
end subroutine
