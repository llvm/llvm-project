# REQUIRES: x86
## Test the difference between the VMA and the LMA for sections with AT().

# RUN: echo '.globl _start; _start: ret; \
# RUN:   .section .a,"a"; .byte 0; \
# RUN:   .section .b,"a"; .byte 0; \
# RUN:   .section .c,"a"; .byte 0; \
# RUN:   .section .d,"a"; .byte 0; \
# RUN:   .data; .byte 0' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t.o
# RUN: ld.lld -T %s %t.o -o %t
# RUN: llvm-readelf -l %t | FileCheck %s

# CHECK:      Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD  0x001000 0x0000000000001000 0x0000000000001000 0x000001 0x000001 R   0x1000

## .b has AT(). It starts a PT_LOAD segment which also includes .c
# CHECK-NEXT: LOAD  0x001001 0x0000000000001001 0x0000000000002005 0x000002 0x000002 R   0x1000

## .d has AT(). It starts a PT_LOAD segment, even if the difference between
## LMA and VMA (0x2007-0x1003) is the same as the previous one.
# CHECK-NEXT: LOAD  0x001003 0x0000000000001003 0x0000000000002007 0x000001 0x000001 R   0x1000

## The orphan section .text starts a PT_LOAD segment. The difference between
## LMA and VMA (0x2008-0x1004) remains the same
# CHECK-NEXT: LOAD  0x001004 0x0000000000001004 0x0000000000002008 0x000001 0x000001 R E 0x1000

## .data starts a PT_LOAD segment. The difference remains the same.
# CHECK-NEXT: LOAD  0x001005 0x0000000000001005 0x0000000000002009 0x000001 0x000001 RW  0x1000

SECTIONS {
  . = 0x1000;
  .a : { *(.a) }
  .b : AT(0x2005) { *(.b) }
  .c : { *(.c) }
  .d : AT(0x2007) { *(.d) }
  ## Orphan section .text will be inserted here.
  .data : { *(.data) }
}
