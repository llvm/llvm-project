## This reproduces a bug where a PIC jump table entry points to a basic block
## that becomes empty after the remove-nops pass.
##
## In relocation mode BOLT creates the jump table object from the PC-relative
## leaq JUMP_TABLE(%rip) reference (BinaryContext::handleAddressRef) whenever
## the referenced memory looks like a PIC jump table. This is independent of
## whether the indirect jump dispatch is recognized. The jump table annotation
## is attached to the jmp only when analyzeIndirectBranch matches the dispatch
## pattern -- and the pattern below defeats it by materializing the jump table
## base twice, into two separate registers. So the jump table object and its
## .rodata entries exist and reference basic blocks, but the jmp has no jump
## table annotation, i.e. BinaryBasicBlock::hasJumpTable() is false for it.
##
## In --strict mode the function is still fully processed: it stays simple with
## the block marked as unknown control flow. Jump table entry 0 targets a
## nop-only block that falls through to the next block. remove-nops empties that
## block, and NormalizeCFG used to redirect its predecessor and delete the empty
## block -- it did not recognize the block as a jump table target because the
## predecessor's hasJumpTable() is false. The jump table object still referenced
## the deleted block by label, leaving a dangling reference in .rodata and
## failing emission with "Undefined temporary symbol".
##
## The fix keeps the block because its predecessor ends in an indirect branch.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -no-pie
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 -v=1 --strict=1 \
# RUN:   -print-cfg -print-only=main 2>&1 | FileCheck %s

## Confirm we exercise the unknown-control-flow / PIC jump table path.
# CHECK: Unknown CF  : true
# CHECK: jmpq {{.*}}%rax # UNKNOWN CONTROL FLOW
# CHECK: PIC Jump table JUMP_TABLE for function main

## Emission must succeed: the (empty) jump table target block is preserved so
## its label stays defined.
# CHECK-NOT: Undefined temporary symbol
# CHECK-NOT: BOLT-ERROR: Emission failed

  .text
  .globl main
  .type main, %function
  .p2align 2
main:
LBB0:
  subq $0x38, %rsp
  cmpl $0x4, %edi
  ja LBBdefault

## Jump table dispatch. Materializing the jump table base twice (into %rax and
## then %rdx) prevents analyzeIndirectBranch from recognizing the pattern, so no
## jump table annotation is attached to the jmp and it stays unknown control
## flow in strict mode.
LBB1:
  movl %edi, %eax
  leaq (,%rax,4), %rdx
  leaq JUMP_TABLE(%rip), %rax
  movl (%rdx,%rax), %eax
  cltq
  leaq JUMP_TABLE(%rip), %rdx
  addq %rdx, %rax
  jmpq *%rax

## Jump table entry 0 target: a single nop that falls through to LBBjoin. The
## remove-nops pass empties this block; NormalizeCFG would (without the fix)
## delete it while the jump table still references it.
LBBnop:
  nop
LBBjoin:
  addq $0x38, %rsp
  ret

LBBc1:
  movl $0x1, %eax
  jmp LBBjoin2
LBBc2:
  movl $0x2, %eax
LBBjoin2:
  addq $0x38, %rsp
  ret

LBBdefault:
  xorl %eax, %eax
  addq $0x38, %rsp
  ret
.size main, .-main

  .rodata
## jump table, entries must be R_X86_64_PC32 relocs
  .globl JUMP_TABLE
JUMP_TABLE:
  .long LBBnop-JUMP_TABLE
  .long LBBc1-JUMP_TABLE
  .long LBBc2-JUMP_TABLE
  .long LBBjoin-JUMP_TABLE
  .long LBBc1-JUMP_TABLE
