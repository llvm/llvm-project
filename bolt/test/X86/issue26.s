# This reproduces issue 26 from our github repo
#  BOLT fails with the following assertion:
#    llvm/tools/llvm-bolt/src/BinaryFunction.cpp:2950: void llvm::bolt::BinaryFunction::postProcessBranches(): Assertion `validateCFG() && "invalid CFG"' failed.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -relocs -print-cfg -o %t.out \
# RUN:    | FileCheck %s

# CHECK-NOT: BOLT-WARNING: CFG invalid in XYZ @ .LBB0

# CHECK: Binary Function "XYZ"

# CHECK: .Ltmp{{.*}} (1 instructions, align : 1)
# CHECK-NEXT: Secondary Entry Point: FUNCat{{.*}}

  .text
  .globl XYZ
  .type XYZ, %function
  .size XYZ, .Lend1-XYZ
XYZ:
  movl %fs:-0x350, %eax
  cmpl %eax, %edi
  jne .L1

  cmp %rdx, (%rsi)
  jne .L2

  movq %rcx, (%rsi)
.L1:
  retq

.L2:
  movl $0xffffffff, %eax
  retq
.Lend1:

  .globl FUNC
  .type FUNC, %function
  .size FUNC, .Lend - FUNC
FUNC:
  cmpq %rdi, %rsi
  je .L1
  retq
.Lend:

  .globl main
  .type main, %function
  .size main, .Lend2 - main
main:
  xorq %rax, %rax
  retq
.Lend2:
