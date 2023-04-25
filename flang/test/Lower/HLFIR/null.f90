! Test lowering of NULL(MOLD) to HLFIR.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s
subroutine test(mold)
  integer, pointer :: mold(:)
  interface
    subroutine takes_ptr(p)
      integer, pointer :: p(:)
    end subroutine
  end interface
  call takes_ptr(null(mold))
end subroutine
! CHECK-LABEL:   func.func @_QPtest(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:  fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:  fir.call @_QPtakes_ptr(%[[VAL_7]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> ()
