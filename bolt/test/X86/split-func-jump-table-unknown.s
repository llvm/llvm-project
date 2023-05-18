# This reproduces a bug with converting an unknown control flow jump table with
# entries pointing to code in function and its cold fragment.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 -v=1 --strict=1 -print-cfg \
# RUN:   -print-only=main 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: marking main.cold.1 as a fragment of main
# CHECK: Binary Function "main" after building cfg
# CHECK: Unknown CF  : true
# CHECK: jmpq    *%r8 # UNKNOWN CONTROL FLOW
# CHECK: PIC Jump table JUMP_TABLE for function main
# CHECK-NEXT: 0x0000 : .Ltmp0
# CHECK-NEXT: 0x0004 : .Ltmp1
# CHECK-NEXT: 0x0008 : __ENTRY_main.cold.1
# CHECK-NEXT: 0x000c : .Ltmp1
  .text
  .globl main
  .type main, %function
  .p2align 2
main:
LBB0:
  leaq JUMP_TABLE(%rip), %r8
  andl $0xf, %ecx
  cmpb $0x4, %cl
  # exit through abort in main.cold.1, registers cold fragment the regular way
  ja main.cold.1

# jump table dispatch, jumping to label indexed by val in %ecx
LBB1:
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
