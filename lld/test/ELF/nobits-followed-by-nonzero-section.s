# REQUIRES: x86
## A non-empty section following .bss must account for .bss's size.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 input.s -o input.o
# RUN: ld.lld -T script.lds input.o -o output
# RUN: llvm-readelf -S output | FileCheck %s

## The .ldata section's file offset must equal .bss's file offset
## plus .bss's size.
# CHECK: .bss     NOBITS   {{[0-9a-f]+}} [[#%.6x,OFF:]] [[#%.6x,SZ:]]
# CHECK-NEXT: .ldata  PROGBITS {{[0-9a-f]+}} [[#%.6x,OFF+SZ]] 000001

#--- input.s

.text
.globl _start
_start:
  ret

.data
.byte 1

.bss
.space 0xfa0

.section .ldata,"aw"
.byte 2

#--- script.lds

SECTIONS {
  .text : {}
  .data : {}
  .bss : {}
  .ldata : { *(.ldata) }
}
