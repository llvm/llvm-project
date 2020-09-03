# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:        .data_noload_a (NOLOAD) : { *(.data_noload_a) } \
# RUN:        .data_noload_b (0x10000) (NOLOAD) : { *(.data_noload_b) } \
# RUN:        .no_input_sec_noload (NOLOAD) : { . += 1; } \
# RUN:        .text (0x20000) : { *(.text) } };" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-readelf -S -l %t | FileCheck %s

# CHECK:      Name                 Type   Address          Off               Size
# CHECK:      .data_noload_a       NOBITS 0000000000000000 [[OFF:[0-9a-f]+]] 001000
# CHECK-NEXT: .data_noload_b       NOBITS 0000000000010000 [[OFF]]           001000
# CHECK-NEXT: .no_input_sec_noload NOBITS 0000000000011000 [[OFF]]           000001

# CHECK:      Type Offset   VirtAddr           PhysAddr
# CHECK-NEXT: LOAD 0x001000 0x0000000000020000 0x0000000000020000

.section .text,"ax",@progbits
  nop

.section .data_noload_a,"aw",@progbits
.zero 4096

.section .data_noload_b,"aw",@progbits
.zero 4096
