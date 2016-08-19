# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld -z stack-size=0x1000 %t -o %t1
# RUN: llvm-readobj -program-headers %t1 | FileCheck %s

.global _start
_start:
  nop

# CHECK:     Type: PT_GNU_STACK (0x6474E551)
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     VirtualAddress: 0x0
# CHECK-NEXT:     PhysicalAddress: 0x0
# CHECK-NEXT:     FileSize: 0
# CHECK-NEXT:     MemSize: 4096
# CHECK-NEXT:     Flags [ (0x6)
# CHECK-NEXT:       PF_R (0x4)
# CHECK-NEXT:       PF_W (0x2)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Alignment: 0
