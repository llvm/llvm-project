! Tests that -finit-local= respects non-default kind mappings.
! Under the default mapping getFKind() happens to equal the byte width, but
! with --kind-mapping overrides the two diverge.  All width calculations must
! use KindMapping::getLogicalBitsize() / getCharacterBitsize() rather than
! getFKind() * 8 or getFKind() directly.
!
! Reproducer: --kind-mapping=l4:8 maps LOGICAL(4) to 8 bits (1 byte).
! Before the fix, the synthesized init emitted a 4-byte i32 constant stored
! via a bitcasted i32* into a 1-byte alloca -- a 3-byte out-of-bounds write.
! After the fix the constant and store are both i8.
!
! RUN: bbc -emit-hlfir --kind-mapping=l4:8 -finit-local=0xAA %s -o - | \
! RUN:     FileCheck --check-prefix=HEX %s
! RUN: bbc -emit-hlfir --kind-mapping=l4:8 -finit-local=zero %s -o - | \
! RUN:     FileCheck --check-prefix=ZERO %s

! LOGICAL(4) remapped to 8 bits -- init must use i8, not i32.
subroutine test_logical4_km(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine

! HEX-LABEL:  func.func @_QPtest_logical4_km(
! HEX:         %[[L:.*]]:2 = hlfir.declare {{.*}}_QFtest_logical4_kmEl
! HEX:         %[[C:.*]] = arith.constant {{.*}} : i8
! HEX:         %[[ADDR:.*]] = fir.convert %[[L]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i8>
! HEX:         fir.store %[[C]] to %[[ADDR]] : !fir.ref<i8>

! ZERO-LABEL: func.func @_QPtest_logical4_km(
! ZERO:        %[[L:.*]]:2 = hlfir.declare {{.*}}_QFtest_logical4_kmEl
! ZERO:        %[[Z:.*]] = fir.zero_bits !fir.logical<4>
! ZERO:        fir.store %[[Z]] to %[[L]]#0 : !fir.ref<!fir.logical<4>>
