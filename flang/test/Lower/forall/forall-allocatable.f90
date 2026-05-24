! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine forall_with_allocatable(a1)
  real :: a1(:)
  real, allocatable :: arr(:)
  forall (i=5:15)
     arr(i) = a1(i)
  end forall
end subroutine forall_with_allocatable

! CHECK-LABEL: func.func @_QPforall_with_allocatable(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a1"}) {
! CHECK:         %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] arg 1 {uniq_name = "_QFforall_with_allocatableEa1"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QFforall_with_allocatableEarr"}
! CHECK:         %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFforall_with_allocatableEarr"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_10:.*]] = arith.constant 15 : i32
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %[[VAL_9]] : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %[[VAL_10]] : i32
! CHECK:         }  (%[[VAL_11:.*]]: i32) {
! CHECK:           %[[VAL_12:.*]] = hlfir.forall_index "i" %[[VAL_11]] : (i32) -> !fir.ref<i32>
! CHECK:           hlfir.region_assign {
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_14]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<f32>
! CHECK:             hlfir.yield %[[VAL_16]] : f32
! CHECK:           } to {
! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:             %[[VAL_20:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_19]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, i64) -> !fir.ref<f32>
! CHECK:             hlfir.yield %[[VAL_20]] : !fir.ref<f32>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }
