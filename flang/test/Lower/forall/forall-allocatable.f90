! Test forall lowering

! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

subroutine forall_with_allocatable(a1)
  real :: a1(:)
  real, allocatable :: arr(:)
  forall (i=5:15)
     arr(i) = a1(i)
  end forall
end subroutine forall_with_allocatable

! CHECK-LABEL: func @_QPforall_with_allocatable(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QFforall_with_allocatableEarr"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFforall_with_allocatableEarr.addr"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.lb0"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.ext0"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 15 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK:         %[[VAL_15:.*]] = fir.shape_shift %[[VAL_12]], %[[VAL_13]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_14]](%[[VAL_15]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_16]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_25]], %[[VAL_22]] : index
! CHECK:           %[[VAL_27:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_26]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_31]], %[[VAL_12]] : index
! CHECK:           %[[VAL_33:.*]] = fir.array_update %[[VAL_20]], %[[VAL_27]], %[[VAL_32]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_33]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_16]], %[[VAL_34:.*]] to %[[VAL_14]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }
