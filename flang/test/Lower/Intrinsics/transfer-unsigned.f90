! RUN: %flang_fc1 -funsigned -emit-hlfir %s -o - | FileCheck %s

! Unsigned integer transfer: arith.bitcast does not support unsigned types,
! so this must use fir.convert on the address instead.
subroutine trans_test_unsigned(store, src)
  ! CHECK-LABEL: func @_QPtrans_test_unsigned(
  ! CHECK:         %[[CAST:.*]] = fir.convert {{.*}} : (!fir.ref<i32>) -> !fir.ref<ui32>
  ! CHECK:         %[[VAL:.*]] = fir.load %[[CAST]] : !fir.ref<ui32>
  ! CHECK-NOT:     arith.bitcast
  ! CHECK-NOT:     fir.call @_FortranATransfer
  ! CHECK:         return
  ! CHECK:       }
  unsigned :: store
  integer :: src
  store = transfer(src, store)
end subroutine
