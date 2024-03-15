# REQUIRES: system-linux

## Check that BOLT correctly parses the Linux kernel .parainstructions section.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie

## Verify paravirtual bindings to instructions.

# RUN: llvm-bolt %t.exe --print-normalized -o %t.out | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 paravirtual patch sites

  .rodata
fptr:
  .quad 0

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
  nop
.L1:
  call *fptr(%rip)
# CHECK:      call
# CHECK-SAME: ParaSite: 1
  nop
.L2:
  call *fptr(%rip)
# CHECK:      call
# CHECK-SAME: ParaSite: 2
  ret
  .size _start, .-_start


## Paravirtual patch sites.
  .section .parainstructions,"a",@progbits

  .balign 8
  .quad .L1      # instruction
  .byte 1        # type
  .byte 7        # length

  .balign 8
  .quad .L2      # instruction
  .byte 1        # type
  .byte 7        # length

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
