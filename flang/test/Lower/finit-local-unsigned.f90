! Tests that -finit-local= preserves the declared unsigned integer type.
! arith.constant only accepts signless integer types, so makeIntCst always
! produces a signless iN constant.  For unsigned locals the result is then
! reinterpreted via fir.convert (iN -> uiN) before the store, which preserves
! the bit pattern and satisfies FIR verification for !fir.ref<ui32>.
! Before the fix, the signless i32 constant was stored directly into
! !fir.ref<ui32>, failing FIR verification with
! "store value type must match memory reference type".
! Zero mode uses fir.zero_bits which is type-polymorphic and was not affected.
!
! RUN: %flang_fc1 -emit-hlfir -funsigned -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX  %s
! RUN: %flang_fc1 -emit-hlfir -funsigned -finit-local=zero  %s -o - | FileCheck --check-prefix=ZERO %s

subroutine test_unsigned(res)
  unsigned :: res
  unsigned :: x
  res = x
end subroutine

! HEX-LABEL:  func.func @_QPtest_unsigned(
! HEX:         %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_unsignedEx
! HEX:         %[[C:.*]] = arith.constant -1431655766 : i32
! HEX:         %[[U:.*]] = fir.convert %[[C]] : (i32) -> ui32
! HEX:         fir.store %[[U]] to %[[X]]#0 : !fir.ref<ui32>

! ZERO-LABEL: func.func @_QPtest_unsigned(
! ZERO:        %[[X:.*]]:2 = hlfir.declare {{.*}}_QFtest_unsignedEx
! ZERO:        %[[Z:.*]] = fir.zero_bits ui32
! ZERO:        fir.store %[[Z]] to %[[X]]#0 : !fir.ref<ui32>
