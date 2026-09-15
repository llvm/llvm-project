# REQUIRES: x86

## A zero-sized section following .bss must not inflate the PT_LOAD p_filesz.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 input.s -o input.o
# RUN: ld.lld -T script.lds input.o -o output
# RUN: llvm-readelf -lW output | FileCheck %s

# CHECK:      Type  Offset  VirtAddr    PhysAddr    FileSiz                MemSiz            Flg  Align
# CHECK:      LOAD  {{.*}}  {{.*}}      {{.*}}  0x[[FILESZ:[0-9a-f]+]] 0x[[MEMSZ:[0-9a-f]+]] RW   {{.*}}
# CHECK-NOT:  [[FILESZ]] == [[MEMSZ]]

#--- input.s

.text
.globl _start
_start:
  ret

.data
.byte 1

.bss
.space 0xfa6

#--- script.lds

SECTIONS {
  .text : {}
  .data : {}
  .bss : {}
  .ldata : { . = ALIGN(8); }
}
