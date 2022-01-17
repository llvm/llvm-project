# This reproduces an issue where two cold fragments are folded into one, so the
# fragment has two parents.
# The fragment is only reachable through a jump table, so all functions must be
# ignored.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -lite=0 -v=1 2>&1 | FileCheck %s

# CHECK-NOT: unclaimed PC-relative relocations left in data
# CHECK-DAG: BOLT-INFO: marking main2.cold.1(*2) as a fragment of main
# CHECK-DAG: BOLT-INFO: marking main2.cold.1(*2) as a fragment of main2
# CHECK-DAG: BOLT-WARNING: Ignoring main2
# CHECK-DAG: BOLT-WARNING: Ignoring main
# CHECK-DAG: BOLT-WARNING: Ignoring main2.cold.1(*2)
# CHECK: BOLT-WARNING: Ignored 3 functions due to cold fragments.
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

  .globl main2
  .type main2, %function
  .p2align 2
main2:
LBB20:
  andl $0xb, %ebx
  cmpb $0x1, %cl
  # exit through ret
  ja LBB23

# jump table dispatch, jumping to label indexed by val in %ecx
LBB21:
  leaq JUMP_TABLE2(%rip), %r8
  movzbl %cl, %ecx
  movslq (%r8,%rcx,4), %rax
  addq %rax, %r8
  jmpq *%r8

LBB22:
  xorq %rax, %rax
LBB23:
  addq $0x8, %rsp
  ret
.size main2, .-main2

# cold fragment is only reachable through jump table
  .globl main2.cold.1
  .type main2.cold.1, %function
main2.cold.1:
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
  .globl JUMP_TABLE1
JUMP_TABLE1:
  .long LBB2-JUMP_TABLE1
  .long LBB3-JUMP_TABLE1
  .long LBB4-JUMP_TABLE1
  .long LBB3-JUMP_TABLE1

  .globl JUMP_TABLE2
JUMP_TABLE2:
  .long LBB22-JUMP_TABLE2
  .long LBB23-JUMP_TABLE2
  .long LBB4-JUMP_TABLE2
  .long LBB23-JUMP_TABLE2
