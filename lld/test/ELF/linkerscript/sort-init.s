# REQUIRES: x86
## Test SORT_BY_INIT_PRIORITY can be used to convert .ctors into .init_array

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/asm -o %t.o
# RUN: ld.lld -T %t/lds %t.o -o %t.out
# RUN: llvm-readelf -x .init_array %t.out | FileCheck %s

# CHECK:      Hex dump of section '.init_array':
# CHECK-NEXT: 0x00000001 00010203 04050607

#--- asm
.globl _start
_start:
  nop

.section foo, "aw", @init_array
  .byte 5

.section .ctors.65435, "a"
  .byte 3
.section .init_array.100, "aw", @init_array
  .byte 4

.section .init_array.7, "aw", @init_array
  .byte 2
.section .ctors.65529,"a"
  .byte 1
.section .init_array.5, "aw", @init_array
  .byte 0

.section .init_array, "aw", @init_array
  .byte 6
.section .ctors, "a"
  .byte 7

#--- lds
SECTIONS {
  .init_array : {
    *(SORT_BY_INIT_PRIORITY(.init_array.* .ctors.*) SORT_BY_INIT_PRIORITY(foo*))
    *(.init_array .ctors)
  }
}
