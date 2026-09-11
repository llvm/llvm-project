! Tests that -finit-local= handles all CHARACTER kind-mapping widths correctly
! using the code unit's allocation stride (alignTo(ceil(charBits/8), ABI)),
! that sub-byte (l4:1) and non-byte-multiple-without-padding (l4:12) LOGICAL
! mappings emit a controlled diagnostic, and that padded LOGICAL mappings
! (allocSize > storeSize, e.g. l4:24 and l4:20) use a byte-fill loop in both
! hex and zero modes.
!
! LOGICAL sub-byte: --kind-mapping=l4:1 maps LOGICAL(4) to 1 bit.
! APInt::getSplat(1, APInt(8, 0xAA)) asserts because the destination width is
! less than 8; a TODO is emitted.
!
! LOGICAL non-byte-multiple (unpadded): --kind-mapping=l4:12 maps LOGICAL(4)
! to 12 bits (storeSize=2, allocSize=2 -- no padding).  makeIntCst(12) would
! produce 0xAAA (i12), storing as AA 0A -- the high nibble unfilled.  A TODO
! is emitted instead.
!
! All CHARACTER kind widths are now handled without diagnostics:
!   a1:1  -- i1 rounds up to i8, stride = 1 byte (fills 1 byte per code unit)
!   a1:12 -- i12 rounds up to i16, stride = 2 bytes
!   a1:24 -- i24 rounds up to i32, stride = 4 bytes (was the motivating case:
!             charBits/8 = 3 missed the last byte; stride = 4 is correct)
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=l4:1 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck %s

! CHECK: not yet implemented: -finit-local= with a sub-byte, non-byte-multiple, or padded LOGICAL kind mapping

subroutine test_logical4_subbyte(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine

! LOGICAL non-byte-multiple: --kind-mapping=l4:12 maps LOGICAL(4) to 12 bits.
! makeIntCst(12) would splat 0xAA into i12 -> 0xAAA, which stores as AA 0A.
! The guard (bits % 8 != 0) catches this and emits a TODO.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=l4:12 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-NONBYTE %s

! LOG-NONBYTE: not yet implemented: -finit-local= with a sub-byte, non-byte-multiple, or padded LOGICAL kind mapping

subroutine test_logical4_nonbyte(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine

! LOGICAL padded mapping: --kind-mapping=l4:24 maps LOGICAL(4) to 24 bits
! (storeSize=3, allocSize=4).  Both hex and zero modes use a byte-fill loop
! over all 4 bytes.
!
! RUN: bbc -emit-hlfir --kind-mapping=l4:24 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-PAD %s

! LOG-PAD-LABEL: func.func @_QPtest_logical4_padded(
! LOG-PAD: %[[C3:.*]] = arith.constant 3 : index
! LOG-PAD: fir.do_loop %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}

subroutine test_logical4_padded(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine

! CHARACTER a1:1: i1 rounds up to i8 (stride = 1 byte).  No diagnostic.
!
! RUN: bbc -emit-hlfir --kind-mapping=a1:1 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=CHAR-1BIT %s

! CHAR-1BIT-NOT: not yet implemented
! CHAR-1BIT: fir.do_loop

subroutine test_char1_subbyte(res)
  character(kind=1, len=2) :: c
  integer :: res
  res = ichar(c(1:1))
end subroutine

! CHARACTER non-byte-multiple cases (a1:12, a1:24): both runs stop at
! test_char1_subbyte, which has the same declarations.  That subroutine
! already covers the % 8 branch of the stride formula (a1:12 gives stride 2,
! a1:24 gives stride 4).  A positive check that no diagnostic fires and a
! fir.do_loop is emitted is sufficient.
!
! RUN: bbc -emit-hlfir --kind-mapping=a1:12 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=CHAR-12BIT %s
! RUN: bbc -emit-hlfir --kind-mapping=a1:24 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=CHAR-24BIT %s

! CHAR-12BIT-NOT: not yet implemented
! CHAR-12BIT: %[[C3:.*]] = arith.constant 3 : index
! CHAR-12BIT: fir.do_loop %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}

! CHAR-24BIT-NOT: not yet implemented
! CHAR-24BIT: %[[C7:.*]] = arith.constant 7 : index
! CHAR-24BIT: fir.do_loop %{{.*}} = %{{.*}} to %[[C7]] step %{{.*}}

! LOGICAL padded mapping, zero mode: same as above with -finit-local=zero.
!
! RUN: bbc -emit-hlfir --kind-mapping=l4:24 -finit-local=zero %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-PAD-ZERO %s

! LOG-PAD-ZERO-LABEL: func.func @_QPtest_logical4_padded(
! LOG-PAD-ZERO: %[[C3:.*]] = arith.constant 3 : index
! LOG-PAD-ZERO: fir.do_loop %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}

! LOGICAL padded non-byte-multiple: --kind-mapping=l4:20 maps LOGICAL(4) to
! 20 bits (storeSize=3, allocSize=4 on most targets since i20 gets 4-byte ABI
! alignment).  Unlike l4:12 (where allocSize==storeSize so the padding guard
! does not fire and a TODO is emitted), l4:20 has allocSize > storeSize, so
! the padding guard fires and both zero and hex modes use a byte-fill loop
! over the full 4-byte allocation.
!
! RUN: bbc -emit-hlfir --kind-mapping=l4:20 -finit-local=zero %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-20-ZERO %s
! RUN: bbc -emit-hlfir --kind-mapping=l4:20 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-20-HEX %s

! LOG-20-ZERO-LABEL: func.func @_QPtest_logical4_padded_nonbyte(
! LOG-20-ZERO: %[[C3:.*]] = arith.constant 3 : index
! LOG-20-ZERO: fir.do_loop %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}

! LOG-20-HEX-LABEL: func.func @_QPtest_logical4_padded_nonbyte(
! LOG-20-HEX: %[[C3:.*]] = arith.constant 3 : index
! LOG-20-HEX: fir.do_loop %{{.*}} = %{{.*}} to %[[C3]] step %{{.*}}

subroutine test_logical4_padded_nonbyte(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine
