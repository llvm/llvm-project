# This reproduces a bug with instrumentation when trying to count calls
# when the target address is computed with a reference to the stack pointer.
# Our instrumentation code uses the stack to save registers to be
# transparent with the instrumented code, but we end up updating the stack
# pointer while doing so, which affects this target address calculation.
# The solution is to temporarily fix RSP. Check that we correctly instrument
# these cases.

# REQUIRES: system-linux,bolt-runtime

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
# RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
# RUN: %t.instrumented arg1 arg2

# Test that the instrumented data makes sense
# RUN:  llvm-bolt %t.exe -o %t.bolted --data %t.fdata \
# RUN:    --reorder-blocks=ext-tsp --reorder-functions=hfsort+ \
# RUN:    --print-only=main --print-finalized | FileCheck %s

# RUN: %t.bolted arg1 arg2

# Check that our indirect call has 1 hit recorded in the fdata file and that
# this was processed correctly by BOLT
# CHECK:         callq   *0x18(%rsp) # CallProfile: 1 (0 misses) :
# CHECK-NEXT:        { targetFunc: 1 (0 misses) }

  .text
  .globl  main
  .type main, %function
  .p2align  4
main:
  pushq %rbp
  movq  %rsp, %rbp
  leaq targetFunc, %rax
  pushq %rax                  # We save the target function address in the stack
  subq  $0x18, %rsp           # Set up a dummy stack frame
  cmpl  $0x2, %edi
  jb    .LBBerror             # Add control flow so we don't have a trivial case
.LBB2:
  callq *0x18(%rsp)           # Indirect call using %rsp
  addq $0x20, %rsp
  movq %rbp, %rsp
  pop %rbp
  retq

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
