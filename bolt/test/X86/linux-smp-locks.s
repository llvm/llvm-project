# REQUIRES: system-linux

## Check that BOLT correctly parses and updates the Linux kernel .smp_locks
## section.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-normalized --keep-nops=0 --bolt-info=0 -o %t.out \
# RUN:   |& FileCheck %s

## Check the output of BOLT with NOPs removed.

# RUN: llvm-bolt %t.out -o %t.out.1 --print-normalized |& FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 SMP lock entries

  .text
  .globl _start
  .type _start, %function
_start:
  nop
  nop
.L0:
  lock incl (%rdi)
# CHECK: lock {{.*}} SMPLock
.L1:
  lock orb $0x40, 0x4(%rsi)
# CHECK: lock {{.*}} SMPLock
  ret
  .size _start, .-_start

  .section .smp_locks,"a",@progbits
  .long .L0 - .
  .long .L1 - .

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
