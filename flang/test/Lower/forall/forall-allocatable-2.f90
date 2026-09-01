! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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

! CHECK-LABEL: func.func @_QPforall_with_allocatable2(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a1"}) {
! CHECK:         %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg 1 {uniq_name = "_QFforall_with_allocatable2Ea1"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "thing", uniq_name = "_QFforall_with_allocatable2Ething"}
! CHECK:         %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFforall_with_allocatable2Ething"} : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
! CHECK:         %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_9]]#0, arr : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_3:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_12:.*]] = arith.constant 15 : i32
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %[[VAL_11]] : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %[[VAL_12]] : i32
! CHECK:         }  (%[[VAL_13:.*]]: i32) {
! CHECK:           %[[VAL_14:.*]] = hlfir.forall_index "i" %[[VAL_13]] : (i32) -> !fir.ref<i32>
! CHECK:           hlfir.region_assign {
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<f32>
! CHECK:             hlfir.yield %[[VAL_18]] : f32
! CHECK:           } to {
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]]#0{"arr"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
! CHECK:             %[[VAL_23:.*]] = hlfir.designate %[[VAL_20]] (%[[VAL_22]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
! CHECK:             hlfir.yield %[[VAL_23]] : !fir.ref<f32>
! CHECK:           }
! CHECK:         }
! CHECK:         %[[VAL_24:.*]] = fir.embox %[[VAL_9]]#0 : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
! CHECK:         fir.call @_FortranADestroy(%[[VAL_25]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:         return
! CHECK:       }
