# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:  . = 0x1000; \
# RUN:  .aaa : { *(.aaa) } \
# RUN:  .bbb : AT(0x2008) { *(.bbb) } \
# RUN:  .ccc : { *(.ccc) } \
# RUN: }" > %t.script
# RUN: ld.lld %t.o --script %t.script -o %t
# RUN: llvm-readelf -l %t | FileCheck %s

# CHECK:      Type  Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: LOAD  0x001000 0x0000000000001000 0x0000000000001000 0x000008 0x000008 R   0x1000
# CHECK-NEXT: LOAD  0x001008 0x0000000000001008 0x0000000000002008 0x000010 0x000010 R   0x1000
# CHECK-NEXT: LOAD  0x001018 0x0000000000001018 0x0000000000001018 0x000001 0x000001 R E 0x1000

.global _start
_start:
 nop

.section .aaa, "a"
.quad 0

.section .bbb, "a"
.quad 0

.section .ccc, "a"
.quad 0
