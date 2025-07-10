! Test lowering of NULL(MOLD) to HLFIR.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
subroutine test1(mold)
  integer, pointer :: mold(:)
  interface
    subroutine takes_ptr(p)
      integer, pointer :: p(:)
    end subroutine
  end interface
  call takes_ptr(null(mold))
end subroutine
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:  fir.call @_QPtakes_ptr(%[[VAL_7]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()

subroutine test2
  integer, pointer :: i
  logical :: l
  l = associated(null(),i)
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2() {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "i", uniq_name = "_QFtest2Ei"}
! CHECK:           %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:           %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest2Ei"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFtest2El"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtest2El"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %false = arith.constant false
! CHECK:           %[[VAL_15:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_6]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           return
! CHECK:         }
