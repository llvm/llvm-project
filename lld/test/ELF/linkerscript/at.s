# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:  . = 0x1000; \
# RUN:  .aaa : AT(0x2000) { *(.aaa) } \
# RUN:  .bbb : { *(.bbb) } \
# RUN:  .ccc : AT(0x3000) { *(.ccc) } \
# RUN:  .ddd : AT(0x4000) { *(.ddd) } \
# RUN:  .eee 0x5000 : AT(0x5000) { *(.eee) } \
# RUN: }" > %t.script
# RUN: ld.lld %t.o --script %t.script -o %t
# RUN: llvm-readelf -l %t | FileCheck %s

# CHECK:      Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD  0x001000 0x0000000000001000 0x0000000000002000 0x000010 0x000010 R   0x1000
# CHECK-NEXT: LOAD  0x001010 0x0000000000001010 0x0000000000003000 0x000008 0x000008 R   0x1000
# CHECK-NEXT: LOAD  0x001018 0x0000000000001018 0x0000000000004000 0x000008 0x000008 R   0x1000
# CHECK-NEXT: LOAD  0x002000 0x0000000000005000 0x0000000000005000 0x000008 0x000008 R   0x1000
# CHECK-NEXT: LOAD  0x002008 0x0000000000005008 0x0000000000005008 0x000001 0x000001 R E 0x1000

.global _start
_start:
 nop

.section .aaa, "a"
.quad 0

.section .bbb, "a"
.quad 0

.section .ccc, "a"
.quad 0

.section .ddd, "a"
.quad 0

.section .eee, "a"
.quad 0
