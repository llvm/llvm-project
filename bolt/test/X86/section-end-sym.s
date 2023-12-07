## Check that BOLT doesn't consider end-of-section symbols (e.g., _etext) as
## functions.

# REQUIRES: x86_64-linux, asserts

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o /dev/null --print-cfg --debug-only=bolt 2>&1 \
# RUN:   | FileCheck %s

# CHECK: considering symbol etext for function
# CHECK-NEXT: rejecting as symbol points to end of its section
# CHECK-NOT: Binary Function "etext{{.*}}" after building cfg


  .text
  .globl _start
  .type _start,@function
_start:
  retq
  .size _start, .-_start

  .align 0x1000
  .globl etext
etext:

  .data
.Lfoo:
  .word 0
