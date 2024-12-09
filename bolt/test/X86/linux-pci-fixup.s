# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-normalized -o %t.out 2>&1 | FileCheck %s

## Check that BOLT correctly parses the Linux kernel .pci_fixup section and
## verify that PCI fixup hook in the middle of a function is detected.

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-WARNING: PCI fixup detected in the middle of function _start
# CHECK:      BOLT-INFO: parsed 2 PCI fixup entries

  .text
  .globl _start
  .type _start, %function
_start:
  nop
.L0:
  ret
  .size _start, .-_start

## PCI fixup table.
  .section .pci_fixup,"a",@progbits

  .short 0x8086     # vendor
  .short 0xbeef     # device
  .long 0xffffffff  # class
  .long 0x0         # class shift
  .long _start - .  # fixup

  .short 0x8086     # vendor
  .short 0xbad      # device
  .long 0xffffffff  # class
  .long 0x0         # class shift
  .long .L0 - .     # fixup

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
