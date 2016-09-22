# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { }" > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readobj -l %t | FileCheck %s

# CHECK:      Type: PT_TLS
# CHECK-NEXT: Offset:
# CHECK-NEXT: VirtualAddress:
# CHECK-NEXT: PhysicalAddress:
# CHECK-NEXT: FileSize: 0
# CHECK-NEXT: MemSize: 9

.section        .tbss,"awT",@nobits
.quad   0
.section        foo,"awT",@nobits
.byte 0
