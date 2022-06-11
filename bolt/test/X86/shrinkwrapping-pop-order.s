# This test reproduces a POP reordering issue in shrink wrapping where we would
# incorrectly put a store after a load (instead of before) when having multiple
# insertions at the same point. Check that the order is correct in this test.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --frame-opt=all --lite=0 \
# RUN:           --print-fop 2>&1 | FileCheck %s

  .globl _start
_start:
    .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 6
    je a
b:  jne _start
# FDATA: 1 _start #b# 1 _start #c# 0 3

c:
  push  %rbx
  push  %rbp
  pop   %rbp
  pop   %rbx

# This basic block is treated as having 0 execution count.
# push and pop will be sinked into this block.
a:
    ud2
    .cfi_endproc


# Check shrink wrapping results:
# CHECK: BOLT-INFO: Shrink wrapping moved 0 spills inserting load/stores and 2 spills inserting push/pops

# Check that order is correct
# CHECK:      Binary Function "_start" after frame-optimizer
# Pushes are ordered according to their reg number and come first
# CHECK:      pushq   %rbp
# CHECK:      pushq   %rbx
# Pops are ordered according to their dominance relation and come last
# CHECK:      popq    %rbx
# CHECK:      popq    %rbp
