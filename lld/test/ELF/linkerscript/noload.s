# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/asm -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/mismatch.s -o %t/mismatch.o
# RUN: ld.lld --script %t/lds %t.o -o %t/out
# RUN: llvm-readelf -S -l %t/out | FileCheck %s

# CHECK:      Name                 Type   Address          Off               Size
# CHECK:      .data_noload_a       NOBITS 0000000000000000 [[OFF:[0-9a-f]+]] 001000
# CHECK-NEXT: .data_noload_b       NOBITS 0000000000010000 [[OFF]]           001000
# CHECK-NEXT: .no_input_sec_noload NOBITS 0000000000011000 [[OFF]]           000001

# CHECK:      Type Offset   VirtAddr           PhysAddr
# CHECK-NEXT: LOAD 0x001000 0x0000000000000000 0x0000000000000000
# CHECK-NEXT: LOAD 0x001000 0x0000000000020000 0x0000000000020000

# CHECK:      00 .data_noload_a .data_noload_b .no_input_sec_noload {{$}}
# CHECK:      01 .text {{$}}

# RUN: not ld.lld --script %t/lds %t.o %t/mismatch.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: section type mismatch for .data_noload_a

#--- asm
.section .text,"ax",@progbits
  nop

.section .data_noload_a,"aw",@nobits
.zero 4096

.section .data_noload_b,"aw",@nobits
.zero 4096

#--- mismatch.s
.section .data_noload_a,"aw",@progbits
.byte 1

#--- lds
SECTIONS {
  .data_noload_a (NOLOAD) : { *(.data_noload_a) }
  .data_noload_b (0x10000) (NOLOAD) : { *(.data_noload_b) }
  .no_input_sec_noload (NOLOAD) : { . += 1; }
  .text (0x20000) : { *(.text) }
}
