# This reproduces an issue where two fragments of same function access same
# jump table, which means at least one fragment visits the other, i.e., one
# of them has split jump table. As a result, all of them will be marked as
# non-simple function.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt -v=3 %t.exe -o %t.out 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: Multiple fragments access same jump table: main; main.cold.1

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
  leaq JUMP_TABLE1(%rip), %r8
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

# cold fragment is only reachable
  .globl main.cold.1
  .type main.cold.1, %function
  .p2align 2
main.cold.1:
  # load bearing nop: pad LBB8 so that it can't be treated
  # as __builtin_unreachable by analyzeJumpTable
  nop
LBB4:
  andl $0xb, %ebx
  cmpb $0x1, %cl
  # exit through ret
  ja LBB7

# jump table dispatch, jumping to label indexed by val in %ecx
LBB5:
  leaq JUMP_TABLE1(%rip), %r8
  movzbl %cl, %ecx
  movslq (%r8,%rcx,4), %rax
  addq %rax, %r8
  jmpq *%r8

LBB6:
  xorq %rax, %rax
LBB7:
  addq $0x8, %rsp
  ret
LBB8:
  callq abort
.size main.cold.1, .-main.cold.1

  .rodata
# jmp table, entries must be R_X86_64_PC32 relocs
  .globl JUMP_TABLE1
JUMP_TABLE1:
  .long LBB2-JUMP_TABLE1
  .long LBB3-JUMP_TABLE1
  .long LBB8-JUMP_TABLE1
  .long LBB6-JUMP_TABLE1
  .long LBB7-JUMP_TABLE1
