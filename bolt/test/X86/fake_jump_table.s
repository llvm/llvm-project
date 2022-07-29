# Currently disassembly is not decoupled from branch target analysis.
# This causes a few checks related to availability of target insn to
# fail for stripped binaries:
#   (a) analyzeJumpTable
#   (b) postProcessEntryPoints
# This test checks if BOLT can safely support instruction bounds check
# for cross-function targets.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -v=1 -print-cfg

  .text
  .globl main
  .type main, %function
  .p2align 2
main:
LBB0:
  .cfi_startproc
  andl $0xf, %ecx
  cmpb $0x4, %cl
  ja .main.cold.1
LBB1:
  leaq FAKE_JUMP_TABLE(%rip), %r8
  cmpq %r8, %r9
LBB2:
  xorq %rax, %rax
  ret
  .cfi_endproc
.size main, .-main

  .globl main.cold.1
  .type main.cold.1, %function
  .p2align 2
main.cold.1:
  .cfi_startproc
  nop
LBB3:
  callq abort
  .cfi_endproc
.size main.cold.1, .-main.cold.1

  .rodata
  .globl FAKE_JUMP_TABLE
FAKE_JUMP_TABLE:
  .long LBB2-FAKE_JUMP_TABLE
  .long LBB3-FAKE_JUMP_TABLE+0x1
