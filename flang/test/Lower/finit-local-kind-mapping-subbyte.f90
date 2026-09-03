! Tests that -finit-local= emits a controlled diagnostic rather than crashing
! when a kind mapping reduces a LOGICAL type to sub-byte storage.
!
! Reproducer: --kind-mapping=l4:1 maps LOGICAL(4) to 1 bit.  Before this fix,
! APInt::getSplat(1, APInt(8, 0xAA)) asserted because the destination width (1)
! is less than the source width (8).  The fix guards this path with a TODO
! diagnostic before reaching getSplat.
!
! RUN: %not_todo_cmd bbc -emit-hlfir --kind-mapping=l4:1 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck %s

! CHECK: not yet implemented: -finit-local= with a sub-byte LOGICAL kind mapping

subroutine test_logical4_subbyte(res)
  logical(kind=4) :: l
  integer :: res
  if (l) res = 1
end subroutine
