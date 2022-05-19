! This test checks lowering of OpenMP DO Directive(Worksharing) with collapse.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

program wsloop_collapse
  integer :: i, j, k
  integer :: a, b, c
  integer :: x
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "b", uniq_name = "_QFEb"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFEj"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFEk"}
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
  a=3
! CHECK:         %[[VAL_7:.*]] = arith.constant 3 : i32
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_0]] : !fir.ref<i32>
  b=2
! CHECK:         %[[VAL_8:.*]] = arith.constant 2 : i32
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i32>
  c=5
! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<i32>
  x=0
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:         fir.store %[[VAL_10]] to %[[VAL_6]] : !fir.ref<i32>

  !$omp do collapse(3)
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_23:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = arith.constant 1 : i32
  do i = 1, a
     do j= 1, b
        do k = 1, c
! CHECK:           omp.wsloop collapse(3) for (%[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]]) : i32 = (%[[VAL_20]], %[[VAL_23]], %[[VAL_26]]) to (%[[VAL_21]], %[[VAL_24]], %[[VAL_27]]) inclusive step (%[[VAL_22]], %[[VAL_25]], %[[VAL_28]]) {
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_9]] : i32
! CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : i32
! CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_11]] : i32
! CHECK:             fir.store %[[VAL_15]] to %[[VAL_6]] : !fir.ref<i32>
! CHECK:             omp.yield
! CHECK:           }
           x = x + i + j + k
        end do
     end do
  end do
  !$omp end do
! CHECK:         return
! CHECK:       }
end program wsloop_collapse
