# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS {.foo : { *(.foo) *(.foo) } }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s
# CHECK:      Sections:
# CHECK-NEXT: Idx Name          Size      Address          Type
# CHECK-NEXT:   0               00000000 0000000000000000
# CHECK-NEXT:   1 .foo          00000004 0000000000000120 DATA
# CHECK-NEXT:   2 .text         00000001 0000000000000124 TEXT DATA

.global _start
_start:
 nop

.section .foo,"a"
 .long 0
