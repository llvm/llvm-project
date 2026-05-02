! This test checks lowering of OpenMP DO Directive(Worksharing) with collapse.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

!CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "WSLOOP_COLLAPSE"} {
program wsloop_collapse
!CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
!CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFEb"}
!CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_10:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
!CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)


!CHECK:           %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
!CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_14:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
!CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_16:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFEk"}
!CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QFEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_18:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK:           %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:           %[[VAL_20:.*]] = arith.constant 3 : i32
!CHECK:           hlfir.assign %[[VAL_20]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>

!CHECK:           %[[VAL_21:.*]] = arith.constant 2 : i32
!CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>

!CHECK:           %[[VAL_22:.*]] = arith.constant 5 : i32
!CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_11]]#0 : i32, !fir.ref<i32>

!CHECK:           %[[VAL_23:.*]] = arith.constant 0 : i32
!CHECK:           hlfir.assign %[[VAL_23]] to %[[VAL_19]]#0 : i32, !fir.ref<i32>

  integer :: i, j, k
  integer :: a, b, c
  integer :: x

  a=3
  b=2
  c=5
  x=0

!CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_26:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_27:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_29:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_30:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_32:.*]] = arith.constant 1 : i32
!CHECK:           omp.wsloop private(@{{.*}} %{{.*}}#0 -> %[[VAL_4:.*]], @{{.*}} %{{.*}}#0 -> %[[VAL_2:.*]], @{{.*}} %{{.*}}#0 -> %[[VAL_0:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT:        omp.loop_nest (%[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]]) : i32 = (%[[VAL_24]], %[[VAL_27]], %[[VAL_30]]) to (%[[VAL_25]], %[[VAL_28]], %[[VAL_31]]) inclusive step (%[[VAL_26]], %[[VAL_29]], %[[VAL_32]]) collapse(3) {
  !$omp do collapse(3)
  do i = 1, a
     do j= 1, b
        do k = 1, c

!CHECK:               %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:               %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:               %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:               hlfir.assign %[[VAL_33]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
!CHECK:               hlfir.assign %[[VAL_34]] to %[[VAL_3]]#0 : i32, !fir.ref<i32>
!CHECK:               hlfir.assign %[[VAL_35]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
!CHECK:               %[[VAL_36:.*]] = fir.load %[[VAL_19]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_37:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_36]], %[[VAL_37]] : i32
!CHECK:               %[[VAL_39:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_38]], %[[VAL_39]] : i32
!CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_40]], %[[VAL_41]] : i32
!CHECK:               hlfir.assign %[[VAL_42]] to %[[VAL_19]]#0 : i32, !fir.ref<i32>
!CHECK:               omp.yield
!CHECK-NEXT:        }
           x = x + i + j + k
        end do
     end do
  end do
!CHECK:           }
  !$omp end do
!CHECK:         return
!CHECK:       }
end program wsloop_collapse
