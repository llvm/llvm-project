## This reproduces a bug with jump table identification where jump table has
## entries pointing to code in function and its cold fragment.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=0 -v=1 --print-cfg --print-only=main \
# RUN:   2>&1 | FileCheck %s

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
  # exit through abort in main.cold.1, registers cold fragment the regular way
  ja main.cold.1

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

# Insert padding between functions, so that the next instruction cannot be
# treated as __builtin_unreachable destination for the jump table.
  .quad 0

  .globl main.cold.1
  .type main.cold.1, %function
  .p2align 2
main.cold.1:
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

## Verify that the entry corresponding to the cold fragment was added to
## the jump table.

# CHECK:      PIC Jump table
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} :
# CHECK-NEXT: 0x{{.*}} : main.cold.1
# CHECK-NEXT: 0x{{.*}} :
