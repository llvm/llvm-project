# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { .text : { *(.text*) } }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s
# CHECK:      .bss          00000004 0000000000000122 BSS
# CHECK-NEXT: .bss          00000100 0000000000000128 BSS

.globl _start
_start:
  jmp _start

.bss
.zero 4

.comm q,128,8
