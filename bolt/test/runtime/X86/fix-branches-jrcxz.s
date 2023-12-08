# Test BOLT does not crash by trying to change the direction of a JRCXZ

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --relocs=1 --reorder-blocks=ext-tsp --print-finalized \
# RUN:    -o %t.out --data %t.fdata | FileCheck %s
# RUN: %t.out 1 2 3

# CHECK: BOLT-INFO

  .text
  .section .text.startup,"ax",@progbits
  .p2align 5,,31
  .globl main
  .type main, %function
main:
  jmp test_function

.globl test_function
.hidden test_function
.type test_function,@function
.align 32
test_function:
# FDATA: 0 main 0 1 test_function 0 0 510
  xorq %rcx, %rcx
  andq $3, %rdi
  jmpq *jumptbl(,%rdi,8)

# Here are the 4 possible targets of the indirect branch to simulate a simple
# CFG. What is important here is that BB1, the first block, conditionally
# transfers control to the exit block with JRCXZ (.J1). We create a mock profile
# saying that this branch is taken more often than not, causing BOLT to try to
# put the exit block after it, which would require us to change the direction
# of JRCXZ.
.BB1:
  movl $0x0, %eax
.J1:
  jrcxz .BBend
# FDATA: 1 test_function #.J1# 1 test_function #.BB2# 0 10
# FDATA: 1 test_function #.J1# 1 test_function #.BBend# 0 500
.BB2:
  movl $0x2, %eax
  jmp .BBend
.Lbb3:
  movl $0x3, %eax
  jmp .BBend
.Lbb4:
  movl $0x4, %eax
.BBend:
  retq
.Lend1:

  .section .rodata
  .globl jumptbl
jumptbl:
  .quad .BB1
  .quad .BB2
  .quad .Lbb3
  .quad .Lbb4
