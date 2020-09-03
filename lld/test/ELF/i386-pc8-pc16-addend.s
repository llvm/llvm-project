# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-gnu %s -o %t1.o

# RUN: ld.lld %t1.o -o %t.out
# RUN: llvm-objdump -s -t %t.out | FileCheck %s
# CHECK: SYMBOL TABLE:
# CHECK: 004010b7 l .und
# CHECK:      Contents of section .text:
# CHECK-NEXT:  4010b4 020000
## 0x4010b7 - 0x4010b4 + addend(-1) = 0x02
## 0x4010b7 - 0x4010b5 + addend(-2) = 0x0000

.byte  und-.-1
.short und-.-2

.section .und, "ax"
und:
