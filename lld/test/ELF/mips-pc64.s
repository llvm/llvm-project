# REQUIRES: mips

# Check handling of 64-bit pc-realtive relocation.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:         .text 0x10000 : { *(.text) } \
# RUN:         .data 0x30000 : { *(.data) } \
# RUN:       }' > %t.script
# RUN: ld.lld -shared %t.o -T %t.script -o %t
# RUN: llvm-readelf -x .data %t | FileCheck %s

# CHECK:      Hex dump of section '.data':
# CHECK-NEXT:  0x00030000 ffffffff fffffff0 00000001 fffdffe8

  .option pic2
  .text
foo:
  nop
  .data
v0:
  .quad foo+0x1fff0-.
v1:
  .quad foo+0x1fffffff0-.
