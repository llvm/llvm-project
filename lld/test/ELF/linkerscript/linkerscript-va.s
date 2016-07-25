# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS {}" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -section-headers %t1 | FileCheck %s
# CHECK:      Sections:
# CHECK-NEXT: Idx Name          Size      Address          Type
# CHECK-NEXT:   0               00000000 0000000000000000
# CHECK-NEXT:   1 .foo          00000004 0000000000000120 DATA
# CHECK-NEXT:   2 .boo          00000004 0000000000000124 DATA
# CHECK-NEXT:   3 .text         00000001 0000000000000128 TEXT DATA

.global _start
_start:
 nop

.section .foo, "a"
foo:
 .long 0

.section .boo, "a"
boo:
 .long 0
