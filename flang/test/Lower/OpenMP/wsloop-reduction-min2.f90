! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! regression test for crash

program reduce
integer :: i = 0
integer :: r = 0

!$omp parallel do reduction(min:r)
do i=0,10
   r = i
enddo
!$omp end parallel do

print *,r

end program

! TODO: the reduction is not curently lowered correctly. This test is checking
! that we do not crash and we still produce the same broken IR as before.

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFEr) : !fir.ref<i32>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFEr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK:             %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_7:.*]] = arith.constant 10 : i32
! CHECK:             %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:             omp.wsloop  for  (%[[VAL_9:.*]]) : i32 = (%[[VAL_6]]) to (%[[VAL_7]]) inclusive step (%[[VAL_8]]) {
! CHECK:               fir.store %[[VAL_9]] to %[[VAL_5]]#1 : !fir.ref<i32>
! CHECK:               %[[VAL_10:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:               hlfir.assign %[[VAL_10]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
