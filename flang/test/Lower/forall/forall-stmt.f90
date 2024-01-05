! Test forall lowering

! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

!*** Test a FORALL statement
subroutine test_forall_stmt(x, mask)

  logical :: mask(200)
  real :: x(200)
  forall (i=1:100,mask(i)) x(i) = 1.
end subroutine test_forall_stmt

! CHECK-LABEL: func @_QPtest_forall_stmt(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<200xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 100 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> !fir.array<200xf32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<200xf32>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_17]] : i64
! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_18]] : (!fir.ref<!fir.array<200x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (!fir.array<200xf32>) {
! CHECK:             %[[VAL_23:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:             %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_25:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:             %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_24]] : index
! CHECK:             %[[VAL_29:.*]] = fir.array_update %[[VAL_13]], %[[VAL_23]], %[[VAL_28]] : (!fir.array<200xf32>, f32, index) -> !fir.array<200xf32>
! CHECK:             fir.result %[[VAL_29]] : !fir.array<200xf32>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_13]] : !fir.array<200xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_30:.*]] : !fir.array<200xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_31:.*]] to %[[VAL_0]] : !fir.array<200xf32>, !fir.array<200xf32>, !fir.ref<!fir.array<200xf32>>
! CHECK:         return
! CHECK:       }
