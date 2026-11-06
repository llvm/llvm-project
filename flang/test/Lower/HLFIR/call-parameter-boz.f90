! RUN: %flang_fc1 -emit-hlfir -Whollerith-or-character-as-boz %s -o - | FileCheck %s

! A CHARACTER named-constant element passed to a scalar INTEGER dummy is
! converted as if it were BOZ (extension); the callee receives the
! character's bits as an integer constant.  'aaaa' is 0x61616161 =
! 1633771873 regardless of endianness.
! CHECK-LABEL: func.func @_QPboz_arg
! CHECK: %[[BITS:.*]] = arith.constant 1633771873 : i32
! CHECK: %[[TMP:.*]]:3 = hlfir.associate %[[BITS]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK: fir.call @_QFboz_argPti(%[[TMP]]#0)
subroutine boz_arg()
  character(4), parameter :: ca(1) = ['aaaa']
  call ti(ca(1))
contains
  subroutine ti(x)
    integer, intent(in) :: x
  end subroutine
end subroutine
