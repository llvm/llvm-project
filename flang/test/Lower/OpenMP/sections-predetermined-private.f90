! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s | FileCheck %s

!$omp parallel sections
!$omp section
    do i = 1, 2
    end do
!$omp section
    do i = 1, 2
    end do
!$omp end parallel sections
end
! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             omp.sections {
! CHECK:               omp.section {
! CHECK:                 %[[VAL_11:.*]]:2 = fir.do_loop %[[VAL_12:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}} -> (index, i32) {
! CHECK:                 }
! CHECK:                 fir.store %[[VAL_11]]#1 to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:                 %[[VAL_25:.*]]:2 = fir.do_loop %[[VAL_26:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {
! CHECK:                 }
! CHECK:                 fir.store %[[VAL_25]]#1 to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
