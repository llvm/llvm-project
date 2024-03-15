# REQUIRES: system-linux

## Check that BOLT correctly parses the Linux kernel __bug_table section.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie

## Verify bug entry bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized -o %t.out | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 bug table entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
  nop
.L0:
  ud2
# CHECK:      ud2
# CHECK-SAME: BugEntry: 1
  nop
.L1:
  ud2
# CHECK:      ud2
# CHECK-SAME: BugEntry: 2
  ret
  .size _start, .-_start


## Bug table.
  .section __bug_table,"a",@progbits
1:
  .long .L0 - .  # instruction
  .org 1b + 12
2:
  .long .L1 - .  # instruction
  .org 2b + 12

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
