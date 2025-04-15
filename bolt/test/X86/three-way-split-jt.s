## This reproduces an issue where the function is split into three fragments
## and all fragments access the same jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -v=1 -print-only=main.warm -print-cfg  2>&1 | FileCheck %s

# CHECK-DAG: BOLT-INFO: marking main.warm as a fragment of main
# CHECK-DAG: BOLT-INFO: marking main.cold as a fragment of main
# CHECK-DAG: BOLT-INFO: processing main.warm as a sibling of non-ignored function
# CHECK-DAG: BOLT-INFO: processing main.cold as a sibling of non-ignored function
# CHECK-DAG: BOLT-WARNING: Ignoring main.cold
# CHECK-DAG: BOLT-WARNING: Ignoring main.warm
# CHECK-DAG: BOLT-WARNING: Ignoring main
# CHECK: BOLT-WARNING: skipped 3 functions due to cold fragments

# CHECK: PIC Jump table JUMP_TABLE for function main, main.warm, main.cold
# CHECK-NEXT:   0x0000 : __ENTRY_main@0x[[#]]
# CHECK-NEXT:   0x0004 : __ENTRY_main@0x[[#]]
# CHECK-NEXT:   0x0008 : __ENTRY_main.cold@0x[[#]]
# CHECK-NEXT:   0x000c : __ENTRY_main@0x[[#]]
  .globl main
  .type main, %function
  .p2align 2
main:
LBB0:
  andl $0xf, %ecx
  cmpb $0x4, %cl
  ## exit through ret
  ja LBB3

## jump table dispatch, jumping to label indexed by val in %ecx
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

  .globl main.warm
  .type main.warm, %function
  .p2align 2
main.warm:
LBB20:
  andl $0xb, %ebx
  cmpb $0x1, %cl
  # exit through ret
  ja LBB23

## jump table dispatch, jumping to label indexed by val in %ecx
LBB21:
  leaq JUMP_TABLE(%rip), %r8
  movzbl %cl, %ecx
  movslq (%r8,%rcx,4), %rax
  addq %rax, %r8
  jmpq *%r8

LBB22:
  xorq %rax, %rax
LBB23:
  addq $0x8, %rsp
  ret
.size main.warm, .-main.warm

## cold fragment is only reachable through jump table
  .globl main.cold
  .type main.cold, %function
main.cold:
  leaq JUMP_TABLE(%rip), %r8
  movzbl %cl, %ecx
  movslq (%r8,%rcx,4), %rax
  addq %rax, %r8
  jmpq *%r8
LBB4:
  callq abort
.size main.cold, .-main.cold

  .rodata
## jmp table, entries must be R_X86_64_PC32 relocs
  .globl JUMP_TABLE
JUMP_TABLE:
  .long LBB2-JUMP_TABLE
  .long LBB3-JUMP_TABLE
  .long LBB4-JUMP_TABLE
  .long LBB3-JUMP_TABLE
