! Test forall lowering

! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

subroutine forall_with_allocatable2(a1)
  real :: a1(:)
  type t
     integer :: i
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing
  forall (i=5:15)
     thing%arr(i) = a1(i)
  end forall
end subroutine forall_with_allocatable2

! CHECK-LABEL: func @_QPforall_with_allocatable2(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "thing", uniq_name = "_QFforall_with_allocatable2Ething"}
! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_4:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_5:.*]] = arith.constant {{.*}} : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) {{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 15 : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = fir.field_index arr, !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_14]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_16]], %[[VAL_17]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_19:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_20:.*]] = fir.shape_shift %[[VAL_18]]#0, %[[VAL_18]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_19]](%[[VAL_20]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_13]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_21]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
! CHECK:           fir.store %[[VAL_26]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:           %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_27]] : index
! CHECK:           %[[VAL_32:.*]] = fir.array_fetch %[[VAL_22]], %[[VAL_31]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_36]], %[[VAL_33]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_update %[[VAL_25]], %[[VAL_32]], %[[VAL_37]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_38]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_39:.*]] to %[[VAL_19]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }
