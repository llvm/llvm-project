! This test checks lowering of OpenMP DO Directive(Worksharing) for different
! types of loop iteration variable, lower bound, upper bound, and step.

!REQUIRES: shell
!RUN: bbc -fopenmp -emit-hlfir %s -o - 2>&1 | FileCheck %s

!CHECK:  OpenMP loop iteration variable cannot have more than 64 bits size and will be narrowed into 64 bits.

program wsloop_variable
  integer(kind=1) :: i1_lb, i1_ub
  integer(kind=2) :: i2, i2_ub, i2_s
  integer(kind=4) :: i4_s
  integer(kind=8) :: i8, i8_s
  integer(kind=16) :: i16, i16_lb
  real :: x

!CHECK:  %[[TMP0:.*]] = arith.constant 1 : i32
!CHECK:  %[[TMP1:.*]] = arith.constant 100 : i32
!CHECK:  %[[TMP2:.*]] = fir.convert %[[TMP0]] : (i32) -> i64
!CHECK:  %[[TMP3:.*]] = fir.convert %{{.*}} : (i8) -> i64
!CHECK:  %[[TMP4:.*]] = fir.convert %{{.*}} : (i16) -> i64
!CHECK:  %[[TMP5:.*]] = fir.convert %{{.*}} : (i128) -> i64
!CHECK:  %[[TMP6:.*]] = fir.convert %[[TMP1]] : (i32) -> i64
!CHECK:  %[[TMP7:.*]] = fir.convert %{{.*}} : (i32) -> i64
!CHECK:  omp.wsloop for (%[[ARG0:.*]], %[[ARG1:.*]]) : i64 = (%[[TMP2]], %[[TMP5]]) to (%[[TMP3]], %[[TMP6]]) inclusive step (%[[TMP4]], %[[TMP7]]) {
!CHECK:    %[[ARG0_I16:.*]] = fir.convert %[[ARG0]] : (i64) -> i16
!CHECK:    fir.store %[[ARG0_I16]] to %[[STORE_IV0:.*]]#1 : !fir.ref<i16>
!CHECK:    fir.store %[[ARG1]] to %[[STORE_IV1:.*]]#1 : !fir.ref<i64>
!CHECK:    %[[LOAD_IV0:.*]] = fir.load %[[STORE_IV0]]#0 : !fir.ref<i16>
!CHECK:    %[[LOAD_IV0_I64:.*]] = fir.convert %[[LOAD_IV0]] : (i16) -> i64
!CHECK:    %[[LOAD_IV1:.*]] = fir.load %[[STORE_IV1]]#0 : !fir.ref<i64>
!CHECK:    %[[TMP10:.*]] = arith.addi %[[LOAD_IV0_I64]], %[[LOAD_IV1]] : i64
!CHECK:    %[[TMP11:.*]] = fir.convert %[[TMP10]] : (i64) -> f32
!CHECK:    hlfir.assign %[[TMP11]] to %{{.*}} : f32, !fir.ref<f32>
!CHECK:    omp.yield
!CHECK:  }

  !$omp do collapse(2)
  do i2 = 1, i1_ub, i2_s
    do i8 = i16_lb, 100, i4_s
      x = i2 + i8
    end do
  end do
  !$omp end do

!CHECK:  %[[TMP12:.*]] = arith.constant 1 : i32
!CHECK:  %[[TMP13:.*]] = fir.convert %{{.*}} : (i8) -> i32
!CHECK:  %[[TMP14:.*]] = fir.convert %{{.*}} : (i64) -> i32
!CHECK:  omp.wsloop for (%[[ARG0:.*]]) : i32 = (%[[TMP12]]) to (%[[TMP13]]) inclusive step (%[[TMP14]])  {
!CHECK:    %[[ARG0_I16:.*]] = fir.convert %[[ARG0]] : (i32) -> i16
!CHECK:    fir.store %[[ARG0_I16]] to %[[STORE3:.*]]#1 : !fir.ref<i16>
!CHECK:    %[[LOAD3:.*]] = fir.load %[[STORE3]]#0 : !fir.ref<i16>
!CHECK:    %[[TMP16:.*]] = fir.convert %[[LOAD3]] : (i16) -> f32
!CHECK:    hlfir.assign %[[TMP16]] to %{{.*}} : f32, !fir.ref<f32>
!CHECK:    omp.yield
!CHECK:  }

  !$omp do
  do i2 = 1, i1_ub, i8_s
    x = i2
  end do
  !$omp end do

!CHECK:  %[[TMP17:.*]] = fir.convert %{{.*}} : (i8) -> i64
!CHECK:  %[[TMP18:.*]] = fir.convert %{{.*}} : (i16) -> i64
!CHECK:  %[[TMP19:.*]] = fir.convert %{{.*}} : (i32) -> i64
!CHECK:  omp.wsloop for (%[[ARG1:.*]]) : i64 = (%[[TMP17]]) to (%[[TMP18]]) inclusive step (%[[TMP19]])  {
!CHECK:    %[[ARG1_I128:.*]] = fir.convert %[[ARG1]] : (i64) -> i128
!CHECK:    fir.store %[[ARG1_I128]] to %[[STORE4:.*]]#1 : !fir.ref<i128>
!CHECK:    %[[LOAD4:.*]] = fir.load %[[STORE4]]#0 : !fir.ref<i128>
!CHECK:    %[[TMP21:.*]] = fir.convert %[[LOAD4]] : (i128) -> f32
!CHECK:    hlfir.assign %[[TMP21]] to %{{.*}} : f32, !fir.ref<f32>
!CHECK:    omp.yield
!CHECK:  }

  !$omp do
  do i16 = i1_lb, i2_ub, i4_s
    x = i16
  end do
  !$omp end do

end program wsloop_variable

!CHECK-LABEL: func.func @_QPwsloop_variable_sub() {
!CHECK:           %[[VAL_0:.*]] = fir.alloca i8 {adapt.valuebyref, pinned}
!CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFwsloop_variable_subEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:           %[[VAL_2:.*]] = fir.alloca i16 {adapt.valuebyref, pinned}
!CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFwsloop_variable_subEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK:           %[[VAL_4:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFwsloop_variable_subEi1"}
!CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFwsloop_variable_subEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:           %[[VAL_6:.*]] = fir.alloca i128 {bindc_name = "i16_lb", uniq_name = "_QFwsloop_variable_subEi16_lb"}
!CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFwsloop_variable_subEi16_lb"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK:           %[[VAL_8:.*]] = fir.alloca i8 {bindc_name = "i1_ub", uniq_name = "_QFwsloop_variable_subEi1_ub"}
!CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFwsloop_variable_subEi1_ub"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:           %[[VAL_10:.*]] = fir.alloca i16 {bindc_name = "i2", uniq_name = "_QFwsloop_variable_subEi2"}
!CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFwsloop_variable_subEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK:           %[[VAL_12:.*]] = fir.alloca i16 {bindc_name = "i2_s", uniq_name = "_QFwsloop_variable_subEi2_s"}
!CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFwsloop_variable_subEi2_s"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK:           %[[VAL_14:.*]] = fir.alloca i32 {bindc_name = "i4_s", uniq_name = "_QFwsloop_variable_subEi4_s"}
!CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFwsloop_variable_subEi4_s"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_16:.*]] = fir.alloca i64 {bindc_name = "i8", uniq_name = "_QFwsloop_variable_subEi8"}
!CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QFwsloop_variable_subEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:           %[[VAL_18:.*]] = fir.alloca i8 {bindc_name = "j1", uniq_name = "_QFwsloop_variable_subEj1"}
!CHECK:           %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]] {uniq_name = "_QFwsloop_variable_subEj1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:           %[[VAL_20:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFwsloop_variable_subEx"}
!CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_20]] {uniq_name = "_QFwsloop_variable_subEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

subroutine wsloop_variable_sub
  integer(kind=1) :: i1, i1_ub, j1
  integer(kind=2) :: i2, i2_s
  integer(kind=4) :: i4_s
  integer(kind=8) :: i8
  integer(kind=16) :: i16_lb
  real :: x

!CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i8>
!CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i16>
!CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (i8) -> i32
!CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (i16) -> i32
!CHECK:           omp.wsloop   for  (%[[VAL_27:.*]]) : i32 = (%[[VAL_22]]) to (%[[VAL_25]]) inclusive step (%[[VAL_26]]) {
!CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i16
!CHECK:             fir.store %[[VAL_28]] to %[[VAL_3]]#1 : !fir.ref<i16>
!CHECK:             %[[VAL_29:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i128>
!CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i128) -> index
!CHECK:             %[[VAL_31:.*]] = arith.constant 100 : i32
!CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
!CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
!CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
!CHECK:             %[[VAL_36:.*]]:2 = fir.do_loop %[[VAL_37:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_34]] iter_args(%[[VAL_38:.*]] = %[[VAL_35]]) -> (index, i64) {
!CHECK:               fir.store %[[VAL_38]] to %[[VAL_17]]#1 : !fir.ref<i64>
!CHECK:               %[[VAL_39:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i16>
!CHECK:               %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i16) -> i64
!CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i64>
!CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_40]], %[[VAL_41]] : i64
!CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i64) -> f32
!CHECK:               hlfir.assign %[[VAL_43]] to %[[VAL_21]]#0 : f32, !fir.ref<f32>
!CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_37]], %[[VAL_34]] : index
!CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
!CHECK:               %[[VAL_46:.*]] = fir.load %[[VAL_17]]#1 : !fir.ref<i64>
!CHECK:               %[[VAL_47:.*]] = arith.addi %[[VAL_46]], %[[VAL_45]] : i64
!CHECK:               fir.result %[[VAL_44]], %[[VAL_47]] : index, i64
!CHECK:             }
!CHECK:             fir.store %[[VAL_48:.*]]#1 to %[[VAL_17]]#1 : !fir.ref<i64>
!CHECK:             omp.yield
!CHECK:           }

  !$omp do
  do i2 = 1, i1_ub, i2_s
    do i8 = i16_lb, 100, i4_s
      x = i2 + i8
    end do
  end do
  !$omp end do


!CHECK:           %[[VAL_49:.*]] = arith.constant 5 : i8
!CHECK:           hlfir.assign %[[VAL_49]] to %[[VAL_19]]#0 : i8, !fir.ref<i8>
!CHECK:           %[[VAL_50:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_51:.*]] = arith.constant 10 : i32
!CHECK:           %[[VAL_52:.*]] = arith.constant 1 : i32
!CHECK:           omp.wsloop   for  (%[[VAL_53:.*]]) : i32 = (%[[VAL_50]]) to (%[[VAL_51]]) inclusive step (%[[VAL_52]]) {
!CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i8
!CHECK:             fir.store %[[VAL_54]] to %[[VAL_1]]#1 : !fir.ref<i8>
!CHECK:             %[[VAL_55:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i8>
!CHECK:             %[[VAL_56:.*]] = fir.load %[[VAL_19]]#0 : !fir.ref<i8>
!CHECK:             %[[VAL_57:.*]] = arith.cmpi eq, %[[VAL_55]], %[[VAL_56]] : i8
!CHECK:             fir.if %[[VAL_57]] {
!CHECK:             } else {
!CHECK:             }
!CHECK:             omp.yield
!CHECK:           }
  j1 = 5
  !$omp do
  do i1 = 1, 10
    if (i1 .eq. j1) then
      print *, "EQ"
    end if
  end do
  !$omp end do

!CHECK:         return
!CHECK:       }

end
