# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64le.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:      _start:
# CHECK-NEXT: 10010004:       1d 00 00 48     bl .+28
# CHECK-NEXT: 10010008:       18 00 41 e8     ld 2, 24(1)
# CHECK-NEXT: 1001000c:       35 00 00 48     bl .+52
# CHECK-NEXT: 10010010:       18 00 41 e8     ld 2, 24(1)

# 0x10010004 + 28 = 0x10010020 (PLT entry 0)
# 0x1001000c + 52 = 0x10010040 (PLT entry 1)

# CHECK:     Disassembly of section .plt:
# CHECK-NEXT: .plt:
# CHECK-NEXT: 10010020:       18 00 41 f8     std 2, 24(1)
# CHECK-NEXT: 10010024:       02 10 82 3d     addis 12, 2, 4098
# CHECK-NEXT: 10010028:       10 80 8c e9     ld 12, -32752(12)
# CHECK-NEXT: 1001002c:       a6 03 89 7d     mtctr 12
# CHECK-NEXT: 10010030:       20 04 80 4e     bctr
# CHECK-NEXT: 10010034:       08 00 e0 7f     trap
# CHECK-NEXT: 10010038:       08 00 e0 7f     trap
# CHECK-NEXT: 1001003c:       08 00 e0 7f     trap
# CHECK-NEXT: 10010040:       18 00 41 f8     std 2, 24(1)
# CHECK-NEXT: 10010044:       02 10 82 3d     addis 12, 2, 4098
# CHECK-NEXT: 10010048:       18 80 8c e9     ld 12, -32744(12)
# CHECK-NEXT: 1001004c:       a6 03 89 7d     mtctr 12
    .text
    .abiversion 2

.type ifunc STT_GNU_IFUNC
.globl ifunc
ifunc:
 nop

.global _start
_start:
  bl foo
  nop
  bl ifunc
  nop
