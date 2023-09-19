! This test checks lowering of OpenMP DO Directive(Worksharing) for different
! types of loop iteration variable, lower bound, upper bound, and step.

!REQUIRES: shell
!RUN: bbc -fopenmp -emit-fir %s -o - 2>&1 | FileCheck %s

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
!CHECK:    fir.store %[[ARG0_I16]] to %[[STORE_IV0:.*]] : !fir.ref<i16>
!CHECK:    fir.store %[[ARG1]] to %[[STORE_IV1:.*]] : !fir.ref<i64>
!CHECK:    %[[LOAD_IV0:.*]] = fir.load %[[STORE_IV0]] : !fir.ref<i16>
!CHECK:    %[[LOAD_IV0_I64:.*]] = fir.convert %[[LOAD_IV0]] : (i16) -> i64
!CHECK:    %[[LOAD_IV1:.*]] = fir.load %[[STORE_IV1]] : !fir.ref<i64>
!CHECK:    %[[TMP10:.*]] = arith.addi %[[LOAD_IV0_I64]], %[[LOAD_IV1]] : i64
!CHECK:    %[[TMP11:.*]] = fir.convert %[[TMP10]] : (i64) -> f32
!CHECK:    fir.store %[[TMP11]] to %{{.*}} : !fir.ref<f32>
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
!CHECK:    fir.store %[[ARG0_I16]] to %[[STORE3:.*]] : !fir.ref<i16>
!CHECK:    %[[LOAD3:.*]] = fir.load %[[STORE3]] : !fir.ref<i16>
!CHECK:    %[[TMP16:.*]] = fir.convert %[[LOAD3]] : (i16) -> f32

!CHECK:    fir.store %[[TMP16]] to %{{.*}} : !fir.ref<f32>
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
!CHECK:    fir.store %[[ARG1_I128]] to %[[STORE4:.*]] : !fir.ref<i128>
!CHECK:    %[[LOAD4:.*]] = fir.load %[[STORE4]] : !fir.ref<i128>
!CHECK:    %[[TMP21:.*]] = fir.convert %[[LOAD4]] : (i128) -> f32
!CHECK:    fir.store %[[TMP21]] to %{{.*}} : !fir.ref<f32>
!CHECK:    omp.yield
!CHECK:  }

  !$omp do
  do i16 = i1_lb, i2_ub, i4_s
    x = i16
  end do
  !$omp end do

end program wsloop_variable

!CHECK-LABEL: func.func @_QPwsloop_variable_sub() {
!CHECK:         %[[IV2:.*]] = fir.alloca i8 {adapt.valuebyref, pinned}
!CHECK:         %[[VAL_0:.*]] = fir.alloca i128 {bindc_name = "i16_lb", uniq_name = "_QFwsloop_variable_subEi16_lb"}
!CHECK:         %[[VAL_1:.*]] = fir.alloca i8 {bindc_name = "i1_ub", uniq_name = "_QFwsloop_variable_subEi1_ub"}
!CHECK:         %[[VAL_2:.*]] = fir.alloca i16 {bindc_name = "i2", uniq_name = "_QFwsloop_variable_subEi2"}
!CHECK:         %[[VAL_3:.*]] = fir.alloca i16 {bindc_name = "i2_s", uniq_name = "_QFwsloop_variable_subEi2_s"}
!CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i4_s", uniq_name = "_QFwsloop_variable_subEi4_s"}
!CHECK:         %[[VAL_5:.*]] = fir.alloca i64 {bindc_name = "i8", uniq_name = "_QFwsloop_variable_subEi8"}
!CHECK:         %[[J1:.*]] = fir.alloca i8 {bindc_name = "j1", uniq_name = "_QFwsloop_variable_subEj1"}
!CHECK:         %[[VAL_6:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFwsloop_variable_subEx"}
!CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i32
!CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i8>
!CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<i16>
!CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (i8) -> i32
!CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (i16) -> i32
!CHECK:         omp.wsloop   for  (%[[ARG0:.*]]) : i32 = (%[[VAL_7]]) to (%[[VAL_10]]) inclusive step (%[[VAL_11]]) {
!CHECK:           %[[ARG0_I16:.*]] = fir.convert %[[ARG0]] : (i32) -> i16
!CHECK:           fir.store %[[ARG0_I16]] to %[[STORE_IV:.*]] : !fir.ref<i16>
!CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<i128>
!CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i128) -> index
!CHECK:           %[[VAL_15:.*]] = arith.constant 100 : i32
!CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
!CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
!CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
!CHECK:           %[[LB:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
!CHECK:           %[[VAL_19:.*]]:2 = fir.do_loop %[[VAL_20:[^ ]*]] =
!CHECK-SAME:          %[[VAL_14]] to %[[VAL_16]] step %[[VAL_18]]
!CHECK-SAME:          iter_args(%[[IV:.*]] = %[[LB]]) -> (index, i64) {
!CHECK:             fir.store %[[IV]] to %[[VAL_5]] : !fir.ref<i64>
!CHECK:             %[[LOAD_IV:.*]] = fir.load %[[STORE_IV]] : !fir.ref<i16>
!CHECK:             %[[VAL_22:.*]] = fir.convert %[[LOAD_IV]] : (i16) -> i64
!CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
!CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_22]], %[[VAL_23]] : i64
!CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> f32
!CHECK:             fir.store %[[VAL_25]] to %[[VAL_6]] : !fir.ref<f32>
!CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_20]], %[[VAL_18]] : index
!CHECK:             %[[STEPCAST:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
!CHECK:             %[[IVLOAD:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
!CHECK:             %[[IVINC:.*]] = arith.addi %[[IVLOAD]], %[[STEPCAST]]
!CHECK:             fir.result %[[VAL_26]], %[[IVINC]] : index, i64
!CHECK:           }
!CHECK:           fir.store %[[VAL_19]]#1 to %[[VAL_5]] : !fir.ref<i64>
!CHECK:           omp.yield
!CHECK:         }

subroutine wsloop_variable_sub
  integer(kind=1) :: i1, i1_ub, j1
  integer(kind=2) :: i2, i2_s
  integer(kind=4) :: i4_s
  integer(kind=8) :: i8
  integer(kind=16) :: i16_lb
  real :: x

  !$omp do
  do i2 = 1, i1_ub, i2_s
    do i8 = i16_lb, 100, i4_s
      x = i2 + i8
    end do
  end do
  !$omp end do

!CHECK:         %[[C1:.*]] = arith.constant 1 : i32
!CHECK:         %[[C10:.*]] = arith.constant 10 : i32
!CHECK:         %[[C1_2:.*]] = arith.constant 1 : i32
!CHECK:         omp.wsloop for (%[[ARG0:.*]]) : i32 = (%[[C1]]) to (%[[C10]]) inclusive step (%[[C1_2]]) {
!CHECK:           %[[ARG0_I8:.*]] = fir.convert %[[ARG0]] : (i32) -> i8
!CHECK:           fir.store %[[ARG0_I8]] to %[[IV2]] : !fir.ref<i8>
!CHECK:           %[[IV2LOAD:.*]] = fir.load %[[IV2]] : !fir.ref<i8>
!CHECK:           %[[J1LOAD:.*]] = fir.load %[[J1]] : !fir.ref<i8>
!CHECK:           %[[VAL_27:.*]] = arith.cmpi eq, %[[IV2LOAD]], %[[J1LOAD]] : i8
!CHECK:           fir.if %[[VAL_27]] {
!CHECK:           } else {
!CHECK:           }
!CHECK:           omp.yield
!CHECK:         }

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
