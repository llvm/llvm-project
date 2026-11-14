! The initial image of an equivalence group holds the bytes in the byte order
! of the target, also when compiling for a big-endian target on a
! little-endian host (and vice versa). The image is built during semantics, so
! the initial values of the globals are already visible in HLFIR.
! REQUIRES: powerpc-registered-target
! RUN: %flang_fc1 -triple powerpc64-unknown-linux-gnu -emit-hlfir %s -o - | FileCheck %s

subroutine chars_over_integers()
  character*4, dimension(2) :: c = (/"0123", "4567"/)
  integer :: k(2)
  equivalence (k, c)
  print *, c(2), k(1)
end subroutine
! "0123" = 0x30313233, "4567" = 0x34353637 in big-endian memory
! CHECK-LABEL: fir.global internal @_QFchars_over_integersEc
! CHECK: arith.constant 808530483 : i32
! CHECK: arith.constant 875902519 : i32
! CHECK: fir.has_value

subroutine integer_over_chars()
  integer :: k = 1
  character*4 :: c
  equivalence (k, c)
  print *, c, k
end subroutine
! The integer is the storage type here: no second byte swap on the way back.
! CHECK-LABEL: fir.global internal @_QFinteger_over_charsEc
! CHECK: arith.constant 1 : i32
! CHECK: fir.has_value
