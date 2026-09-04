! Tests that -finit-local= emits a controlled diagnostic rather than crashing
! or silently misbehaving when a kind mapping produces a sub-byte or
! non-byte-multiple width for LOGICAL or CHARACTER types.
!
! LOGICAL reproducer: --kind-mapping=l4:1 maps LOGICAL(4) to 1 bit.  Before
! this fix, APInt::getSplat(1, APInt(8, 0xAA)) asserted because the destination
! width (1) is less than the source width (8).
!
! CHARACTER reproducer: --kind-mapping=a1:1 maps CHARACTER(1) to 1 bit.
! getCharacterBitsize / 8 truncates to zero, silently skipping initialization.
! --kind-mapping=a1:12 produces a 12-bit width; integer division gives
! kindBytes=1, so only half the bytes would be covered.
! Both cases now emit a controlled TODO diagnostic.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=l4:1 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck %s

! CHECK: not yet implemented: -finit-local= with a sub-byte LOGICAL kind mapping

subroutine test_logical4_subbyte(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine

! CHARACTER kind-mapping sub-byte case: --kind-mapping=a1:1 maps CHARACTER(1)
! to 1 bit.  getCharacterBitsize(1) / 8 would truncate to zero, silently
! producing no initialization.  The guard emits a controlled diagnostic instead.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=a1:1 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=CHAR-SUBBYTE %s

! CHAR-SUBBYTE: not yet implemented: -finit-local= with a sub-byte or non-byte-multiple CHARACTER kind mapping

subroutine test_char1_subbyte(res)
  character(kind=1, len=2) :: c
  integer :: res
  res = ichar(c(1:1))
end subroutine

! CHARACTER kind-mapping non-byte-multiple case: --kind-mapping=a1:12 maps
! CHARACTER(1) to 12 bits.  getCharacterBitsize(1) / 8 = 1, so the loop
! would run nUnits times with a 1-byte stride, covering only half the storage.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=a1:12 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=CHAR-NONBYTE %s

! CHAR-NONBYTE: not yet implemented: -finit-local= with a sub-byte or non-byte-multiple CHARACTER kind mapping

subroutine test_char1_nonbyte(res)
  character(kind=1, len=2) :: c
  integer :: res
  res = ichar(c(1:1))
end subroutine
