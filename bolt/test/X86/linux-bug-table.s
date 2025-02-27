# REQUIRES: system-linux

## Check that BOLT correctly parses and updates the Linux kernel __bug_table
## section.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie

## Verify bug entry bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized --print-only=_start -o %t.out \
# RUN:   --eliminate-unreachable=1 --bolt-info=0 | FileCheck %s

## Verify bug entry bindings again after unreachable code elimination.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-only=_start --print-normalized \
# RUN:   2>&1 | FileCheck --check-prefix=CHECK-REOPT %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 bug table entries

  .text
  .globl _start
  .type _start, %function
_start:
  jmp .L1
.L0:
  ud2
# CHECK:      ud2
# CHECK-SAME: BugEntry: 1
.L1:
  ud2
# CHECK:      ud2
# CHECK-SAME: BugEntry: 2

## Only the second entry should remain after the first pass.

# CHECK-REOPT: ud2
# CHECK-REOPT-SAME: BugEntry: 2

  ret
## The return instruction is reachable only via preceding ud2. Test that it is
## treated as a reachable instruction in the Linux kernel mode.

# CHECK-REOPT-NEXT: ret
  .size _start, .-_start


## Bug table.
  .section __bug_table,"a",@progbits
1:
  .long .L0 - .  # instruction
  .org 1b + 12
2:
  .long .L1 - .  # instruction
  .org 2b + 12

## Linux kernel version
  .rodata
  .align 16
  .globl linux_banner
  .type  linux_banner, @object
linux_banner:
  .string  "Linux version 6.6.61\n"
  .size  linux_banner, . - linux_banner

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
