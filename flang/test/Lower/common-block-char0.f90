! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test that CHARACTER*0 symbols in a COMMON block are assigned their correct
! sequential offset rather than always offset 0 (the block base address).
!
! Block /blk/ layout:
!   offset  0 : ii1  (integer(8), 8 bytes)
!   offset  8 : zc0  (character*0, 0 bytes)  <- zero-size, no storage consumed
!   offset  8 : ll1  (integer(8), 8 bytes)
!   Total: 16 bytes
!
! Before fix (compute-offsets.cpp): DoSymbol() returned early for zero-size
! symbols without calling symbol.set_offset(), leaving zc0 stamped with
! offset 0. This caused storage(%1[0]) for zc0 instead of storage(%1[8]).

subroutine char0_common
  integer(8) :: ii1
  character*0 :: zc0
  integer(8) :: ll1
  common /blk/ ii1, zc0, ll1
  call use(ii1, zc0, ll1)
end subroutine

! CHECK: fir.global common @blk_(dense<0> : vector<16xi8>) {alignment = 8 : i64} : !fir.array<16xi8>

! CHECK-LABEL: func.func @_QPchar0_common

! ii1 at offset 0
! CHECK: %[[BASE:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<16xi8>>
! CHECK: hlfir.declare {{.*}} storage(%[[BASE]][0]) {uniq_name = "_QFchar0_commonEii1"}

! ll1 at offset 8
! CHECK: hlfir.declare {{.*}} storage(%[[BASE]][8]) {uniq_name = "_QFchar0_commonEll1"}

! zc0 at offset 8 (not 0) -- key assertion
! CHECK: hlfir.declare {{.*}} storage(%[[BASE]][8]) {uniq_name = "_QFchar0_commonEzc0"}
