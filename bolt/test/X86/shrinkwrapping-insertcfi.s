# This test reproduces the issue with inserting updated CFI in shrink wrapping
# into the first basic block.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --frame-opt=all --lite=0 \
# RUN:           --print-fop 2>&1 | FileCheck %s

# Check shrink wrapping results:
# CHECK: BOLT-INFO: Shrink wrapping moved 0 spills inserting load/stores and 1 spills inserting push/pops

# Check that CFI is successfully inserted into the first basic block:
# CHECK:      Binary Function "_start" after frame-optimizer
# CHECK:      .LBB00 (2 instructions, align : 1)
# CHECK-NEXT: Entry Point
# CHECK:      00000000:   !CFI {{.*}}
# CHECK-NEXT: 00000000:   je  .Ltmp{{.*}}

  .globl _start
_start:
    .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 6
# !CFI OpOffset for reg3/rbx is inserted into this block.
    je	a
b:  jne _start
# FDATA: 1 _start #b# 1 _start #c# 0 3

c:
    push	%rbx
    .cfi_offset 3, 4
    pop	%rbx

# This basic block is treated as having 0 execution count.
# push and pop will be sinked into this block.
a:
    ud2
    .cfi_endproc
