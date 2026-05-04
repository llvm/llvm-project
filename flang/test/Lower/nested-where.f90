! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QQmain() attributes {fir.bindc_name = "NESTED_WHERE"} {
program nested_where

  ! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<3xi32>>
  ! CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  ! CHECK:  %[[VAL_5:.*]] = fir.address_of(@_QFEmask1) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_7]]) {uniq_name = "_QFEmask1"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  ! CHECK:  %[[VAL_9:.*]] = fir.address_of(@_QFEmask2) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:  %[[VAL_10:.*]] = arith.constant 3 : index
  ! CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:  %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_11]]) {uniq_name = "_QFEmask2"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  ! CHECK:  %[[VAL_13:.*]] = arith.constant 1 : i32
  ! CHECK:  %[[VAL_14:.*]] = arith.constant 3 : i32
  ! CHECK:  hlfir.forall lb {
  ! CHECK:    hlfir.yield %[[VAL_13]] : i32
  ! CHECK:  } ub {
  ! CHECK:    hlfir.yield %[[VAL_14]] : i32
  ! CHECK:  }  (%[[VAL_15:.*]]: i32) {
  ! CHECK:    hlfir.where {
  ! CHECK:      hlfir.yield %[[VAL_8]]#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:    } do {
  ! CHECK:      hlfir.where {
  ! CHECK:        hlfir.yield %[[VAL_12]]#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:      } do {
  ! CHECK:        hlfir.region_assign {
  ! CHECK:          %[[VAL_16:.*]] = arith.constant 1 : i32
  ! CHECK:          hlfir.yield %[[VAL_16]] : i32
  ! CHECK:        } to {
  ! CHECK:          hlfir.yield %[[VAL_4]]#0 : !fir.ref<!fir.array<3xi32>>
  ! CHECK:        }
  ! CHECK:      }
  ! CHECK:    }
  ! CHECK:  }

  integer :: a(3) = 0
  logical :: mask1(3) = (/ .true.,.false.,.true. /)
  logical :: mask2(3) = (/ .true.,.true.,.false. /)
  forall (i=1:3)
    where (mask1)
      where (mask2)
        a = 1
      end where
    endwhere
  end forall
  ! CHECK:  return
  ! CHECK: }
end program nested_where
