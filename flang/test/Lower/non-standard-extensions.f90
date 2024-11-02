! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of non standard features.

! Test mismatch on result type between callee/caller
! CHECK-LABEL: func @_QPexpect_i32
subroutine expect_i32()
  external :: returns_i32
  real(4) :: returns_i32
  ! CHECK: %[[funcAddr:.*]] = fir.address_of(@_QPreturns_i32) : () -> i32
  ! CHECK: %[[funcCast:.*]] = fir.convert %[[funcAddr]] : (() -> i32) -> (() -> f32)
  ! CHECK: fir.call %[[funcCast]]() : () -> f32
  print *, returns_i32()
end subroutine
integer(4) function returns_i32()
end function
