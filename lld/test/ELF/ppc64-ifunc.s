# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:      _start:
# CHECK-NEXT:  10010004: {{.*}} bl .+12
# CHECK-NEXT:  10010008: {{.*}} bl .+40

# 0x10010004 + 12 = 0x10010010 (PLT entry 0)
# 0x10010008 + 40 = 0x10010030 (PLT entry 1)

# CHECK:     Disassembly of section .plt:
# CHECK:       10010010: {{.*}} std 2, 40(1)
# CHECK-NEXT:  10010014: {{.*}} addis 11, 2, 4098
# CHECK-NEXT:  10010018: {{.*}} ld 12, -32744(11)
# CHECK-NEXT:  1001001c: {{.*}} ld 11, 0(12)
# CHECK-NEXT:  10010020: {{.*}} mtctr 11
# CHECK-NEXT:  10010024: {{.*}} ld 2, 8(12)
# CHECK-NEXT:  10010028: {{.*}} ld 11, 16(12)
# CHECK-NEXT:  1001002c: {{.*}} bctr
# CHECK-NEXT:  10010030: {{.*}} std 2, 40(1)
# CHECK-NEXT:  10010034: {{.*}} addis 11, 2, 4098
# CHECK-NEXT:  10010038: {{.*}} ld 12, -32736(11)
# CHECK-NEXT:  1001003c: {{.*}} ld 11, 0(12)
# CHECK-NEXT:  10010040: {{.*}} mtctr 11
# CHECK-NEXT:  10010044: {{.*}} ld 2, 8(12)
# CHECK-NEXT:  10010048: {{.*}} ld 11, 16(12)
# CHECK-NEXT:  1001004c: {{.*}} bctr

.type ifunc STT_GNU_IFUNC
.globl ifunc
ifunc:
 nop

.global _start
_start:
  bl bar
  bl ifunc
