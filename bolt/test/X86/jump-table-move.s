# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 -v=1 --jump-tables=move 2>&1 | FileCheck %s

# CHECK-NOT: unclaimed PC-relative relocations left in data
# CHECK: BOLT-INFO: marking main.cold.1 as a fragment of main
# CHECK: BOLT-WARNING: Ignoring main.cold.1
# CHECK: BOLT-WARNING: Ignoring main

  .text
  .globl main
  .type main, %function
  .p2align 2
main:
    cmpl  $0x67, %edi
    jne main.cold.1
LBB0:
    retq
.size main, .-main

  .globl main.cold.1
  .type main.cold.1, %function
  .p2align 2
main.cold.1:
    jmpq  *JUMP_TABLE(,%rcx,8)
.size main.cold.1, .-main.cold.1

  .rodata
  .globl JUMP_TABLE
JUMP_TABLE:
  .quad LBB0
  .quad LBB0
