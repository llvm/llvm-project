# REQUIRES: x86
## A non-empty section following .bss must account for .bss's size.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:   .text : {} \
# RUN:   .data : {} \
# RUN:   .bss : {} \
# RUN:   .ldata : { *(.ldata) } \
# RUN: }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

## The .ldata section's file offset must equal .bss's file offset
## plus .bss's size.
# CHECK: .bss     NOBITS   {{[0-9a-f]+}} [[#%.6x,OFF:]] [[#%.6x,SZ:]]
# CHECK-NEXT: .ldata  PROGBITS {{[0-9a-f]+}} [[#%.6x,OFF+SZ]] 000001

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
