# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
# RUN: echo "SECTIONS {                                        \
# RUN:         .ARM.exidx 0x1000 : { *(.ARM.exidx*) foo = .; } \
# RUN:         .text      0x2000 : { *(.text*) }               \
# RUN:       }" > %t.script
## We used to crash if the last output section command for .ARM.exidx
## was anything but an input section description.
# RUN: ld.lld --no-merge-exidx-entries -T %t.script %t.o -shared -o %t.so
# RUN: llvm-objdump -s -triple=armv7a-none-linux-gnueabi %t.so | FileCheck %s

 .syntax unified
 .text
 .global _start
_start:
 .fnstart
 .cantunwind
 bx lr
 .fnend

# CHECK: Contents of section .ARM.exidx:
# 1000 + 1000 = 0x2000 = _start
# 1008 + 0ffc = 0x2004 = _start + sizeof(_start)
# CHECK-NEXT: 1000 00100000 01000000 fc0f0000 01000000
