# This reproduces a bug with instrumentation when trying to instrument
# a function with only tail calls. Such functions can clobber red zone,
# see https://github.com/llvm/llvm-project/issues/61114.

# REQUIRES: system-linux,bolt-runtime

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
# RUN:   -o %t.instrumented
# RUN: %t.instrumented arg1 arg2
# RUN: llvm-objdump %t.instrumented --disassemble-symbols=main | FileCheck %s

# CHECK: leaq 0x80(%rsp), %rsp

  .text
  .globl  main
  .type main, %function
  .p2align  4
main:
  pushq %rbp
  movq  %rsp, %rbp
  mov   %rax,-0x10(%rsp)
  leaq targetFunc, %rax
  pushq %rax                  # We save the target function address in the stack
  subq  $0x18, %rsp           # Set up a dummy stack frame
  cmpl  $0x2, %edi
  jb    .LBBerror             # Add control flow so we don't have a trivial case
.LBB2:
  addq $0x20, %rsp
  movq %rbp, %rsp
  pop %rbp
  mov -0x10(%rsp),%rax
  jmp targetFunc

.LBBerror:
  addq $0x20, %rsp
  movq %rbp, %rsp
  pop %rbp
  movq $1, %rax               # Finish with an error if we go this path
  retq
  .size main, .-main

  .globl targetFunc
  .type targetFunc, %function
  .p2align  4
targetFunc:
  xorq %rax, %rax
  retq
  .size targetFunc, .-targetFunc
