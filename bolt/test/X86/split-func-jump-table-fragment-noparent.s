# This reproduces a bug with jump table identification where jump table has
# entries pointing to code in function and its cold fragment.
# The fragment is only reachable through jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -lite=0 -v=1 2>&1 | FileCheck %s

# CHECK-NOT: unclaimed PC-relative relocations left in data
# CHECK: BOLT-INFO: marking main.cold.1 as a fragment of main
  .text
  .globl main
  .type main, %function
  .p2align 2
main:
LBB0:
  andl $0xf, %ecx
  cmpb $0x4, %cl
  # exit through ret
  ja LBB3

# jump table dispatch, jumping to label indexed by val in %ecx
LBB1:
  leaq JUMP_TABLE(%rip), %r8
  movzbl %cl, %ecx
  movslq (%r8,%rcx,4), %rax
  addq %rax, %r8
  jmpq *%r8

LBB2:
  xorq %rax, %rax
LBB3:
  addq $0x8, %rsp
  ret
.size main, .-main

# cold fragment is only reachable through jump table
  .globl main.cold.1
  .type main.cold.1, %function
  .p2align 2
main.cold.1:
  # load bearing nop: pad LBB4 so that it can't be treated
  # as __builtin_unreachable by analyzeJumpTable
  nop
LBB4:
  callq abort
.size main.cold.1, .-main.cold.1

  .rodata
# jmp table, entries must be R_X86_64_PC32 relocs
  .globl JUMP_TABLE
JUMP_TABLE:
  .long LBB2-JUMP_TABLE
  .long LBB3-JUMP_TABLE
  .long LBB4-JUMP_TABLE
  .long LBB3-JUMP_TABLE
