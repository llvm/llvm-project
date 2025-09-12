! RUN: bbc --strict-fir-volatile-verifier -fopenmp %s -o - | FileCheck %s
program main
implicit none
integer,volatile::a
integer::n,i
a=0
n=1000
!$omp parallel 
!$omp do reduction(+:a)
  do i=1,n
    a=a+1
  end do
!$omp end parallel
end program

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "MAIN"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant 1000 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
! CHECK:           %[[VAL_5:.*]] = fir.volatile_cast %[[VAL_4]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEa"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_9:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_6]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[VAL_1]] to %[[VAL_10]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
! CHECK:             omp.wsloop private(@_QFEi_private_i32 %[[VAL_8]]#0 -> %[[VAL_12:.*]] : !fir.ref<i32>) reduction(@add_reduction_i32 %[[VAL_6]]#0 -> %[[VAL_13:.*]] : !fir.ref<i32, volatile>) {
! CHECK:               omp.loop_nest (%[[VAL_14:.*]]) : i32 = (%[[VAL_0]]) to (%[[VAL_11]]) inclusive step (%[[VAL_0]]) {
! CHECK:                 %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                 %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_13]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEa"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:                 hlfir.assign %[[VAL_14]] to %[[VAL_15]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_17:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<i32, volatile>
! CHECK:                 %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_0]] : i32
! CHECK:                 hlfir.assign %[[VAL_18]] to %[[VAL_16]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
