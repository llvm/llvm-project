## Regression test for a crash in StackLayoutModifier when a 1-byte stack
## access is located in a stack region moved by shrink wrapping.
##
## Shrink wrapping moves the push of %rbx from the hot block "c" to the
## cold block "a". StackLayoutModifier then has to adjust all stack
## accesses affected by the collapsed region, including byte-sized ones
## (movb).

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --frame-opt=all --lite=0 \
# RUN:           --print-fop 2>&1 | FileCheck %s

  .globl _start
  .type _start, %function
_start:
    .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 6
    je a
b:  jne b
# FDATA: 1 _start #b# 1 _start #b# 0 3
# FDATA: 1 _start #b# 1 _start #c# 0 3
c:
  push  %rbx
  movb  %al, 0x10(%rsp)
  movb  0x10(%rsp), %cl
  pop   %rbx
  ret
## This basic block is treated as having 0 execution count.
## The push/pop will be sinked into this block, collapsing the region
## in the hot block and requiring byte accesses to be readjusted.
a:
  ud2
    .cfi_endproc
  .size _start, .-_start

# CHECK: BOLT-INFO: Shrink wrapping moved 0 spills inserting load/stores and 1 spills inserting push/pops
# CHECK: Binary Function "_start" after frame-optimizer
# CHECK: movb %al, 0x8(%rsp)
# CHECK: movb 0x8(%rsp), %cl
