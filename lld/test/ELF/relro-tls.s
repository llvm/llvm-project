# REQUIRES: x86

## PT_GNU_RELRO includes TLS sections.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -l %t | FileCheck %s

## Currently p_memsz of PT_GNU_RELRO is rounded up to protect the last page.

# CHECK:      Type: PT_GNU_RELRO
# CHECK:      VirtualAddress: 0x2021C8
# CHECK:      FileSize: 4
# CHECK-NEXT: MemSize: 3640
# CHECK:      Alignment: 1

.section .foo,"awT",@progbits
.long 1

.section .bar,"awT",@nobits
.space 2
