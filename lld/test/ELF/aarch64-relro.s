# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-readobj -l %t2 | FileCheck %s

# CHECK:      Type: PT_GNU_RELRO
# CHECK-NEXT: Offset:
# CHECK-NEXT: VirtualAddress: 0x220190
# CHECK-NEXT: PhysicalAddress:
# CHECK-NEXT: FileSize:
# CHECK-NEXT: MemSize: 3696

.section .data.rel.ro,"aw",%progbits
.byte 1
