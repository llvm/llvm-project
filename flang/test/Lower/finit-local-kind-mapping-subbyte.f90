! Tests that -finit-local= handles all CHARACTER kind-mapping widths correctly
! using the code unit's allocation stride (alignTo(ceil(charBits/8), ABI)),
! and that LOGICAL sub-byte and non-byte-multiple mappings emit a controlled
! diagnostic.
!
! LOGICAL sub-byte: --kind-mapping=l4:1 maps LOGICAL(4) to 1 bit.
! APInt::getSplat(1, APInt(8, 0xAA)) asserts because the destination width is
! less than 8; a TODO is emitted.
!
! LOGICAL non-byte-multiple: --kind-mapping=l4:12 maps LOGICAL(4) to 12 bits.
! makeIntCst(12) would produce 0xAAA (i12), which stores as AA 0A -- the high
! nibble of the second byte is not filled.  A TODO is emitted instead.
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

! LOGICAL padded mapping: --kind-mapping=l4:24 maps LOGICAL(4) to 24 bits.
! An i24 has a 4-byte allocation size (storeSize=3, allocSize=4). A 3-byte store
! would leave the 4th byte unwritten. The guard catches this and emits a TODO.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=l4:24 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=LOG-PAD %s

! LOG-PAD: not yet implemented: -finit-local= with a sub-byte, non-byte-multiple, or padded LOGICAL kind mapping

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
