! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.declare_reduction @add_reduction_byref_box_Uxf32 : !fir.ref<!fir.box<!fir.array<?xf32>>> alloc {
!                 [...]
! CHECK:          omp.yield
! CHECK-LABEL:  } init {
!                 [...]
! CHECK:          omp.yield
! CHECK-LABEL:  } combiner {
!                 [...]
! CHECK:          omp.yield
! CHECK-LABEL:  }  cleanup {
!                  [...]
! CHECK:           omp.yield
! CHECK:  }

! CHECK-LABEL:  func.func @_QPtask_reduction
! CHECK-SAME:  (%[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:          %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:          %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]]
! CHECK-SAME:      {uniq_name = "_QFtask_reductionEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:          omp.parallel {
! CHECK:            %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:            fir.store %[[VAL_2]]#1 to %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:            omp.taskgroup task_reduction(byref @add_reduction_byref_box_Uxf32 %[[VAL_3]] -> %[[VAL_4:.*]]: !fir.ref<!fir.box<!fir.array<?xf32>>>) {
! CHECK:              %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] 
! CHECK-SAME:         {uniq_name = "_QFtask_reductionEx"} : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> (!fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.array<?xf32>>>)
! CHECK:              omp.task in_reduction(byref @add_reduction_byref_box_Uxf32 %[[VAL_5]]#0 -> %[[VAL_6:.*]] : !fir.ref<!fir.box<!fir.array<?xf32>>>) {
!                       [...]
! CHECK:                omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:              }
! CHECK:              omp.terminator
! CHECK:             }
! CHECK:             return
! CHECK:           }

subroutine task_reduction(x)
   real, dimension(:) :: x
   !$omp parallel
   !$omp taskgroup task_reduction(+:x)
   !$omp task in_reduction(+:x)
   x = x + 1
   !$omp end task
   !$omp end taskgroup
   !$omp end parallel
end subroutine
