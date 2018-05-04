# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:      _start:
# CHECK-NEXT: 10010004:       {{.*}}     bl .+28
# CHECK-NEXT: 10010008:       {{.*}}     ld 2, 24(1)
# CHECK-NEXT: 1001000c:       {{.*}}     bl .+52
# CHECK-NEXT: 10010010:       {{.*}}     ld 2, 24(1)

# 0x10010004 + 28 = 0x10010020 (PLT entry 0)
# 0x1001000c + 52 = 0x10010040 (PLT entry 1)

# CHECK:     Disassembly of section .plt:
# CHECK-NEXT: .plt:
# CHECK-NEXT: 10010020:       {{.*}}     std 2, 24(1)
# CHECK-NEXT: 10010024:       {{.*}}     addis 12, 2, 4098
# CHECK-NEXT: 10010028:       {{.*}}     ld 12, -32752(12)
# CHECK-NEXT: 1001002c:       {{.*}}     mtctr 12
# CHECK-NEXT: 10010030:       {{.*}}     bctr
# CHECK-NEXT: 10010034:       {{.*}}     trap
# CHECK-NEXT: 10010038:       {{.*}}     trap
# CHECK-NEXT: 1001003c:       {{.*}}     trap
# CHECK-NEXT: 10010040:       {{.*}}     std 2, 24(1)
# CHECK-NEXT: 10010044:       {{.*}}     addis 12, 2, 4098
# CHECK-NEXT: 10010048:       {{.*}}     ld 12, -32744(12)
# CHECK-NEXT: 1001004c:       {{.*}}     mtctr 12
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
