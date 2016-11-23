# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
# RUN: ld.lld %t -o %t.out
# RUN: llvm-readobj --program-headers %t.out | FileCheck %s

# CHECK:      ProgramHeader {
# CHECK:        Type: PT_OPENBSD_RANDOMIZE (0x65A3DBE6)
# CHECK-NEXT:   Offset: 0x158
# CHECK-NEXT:   VirtualAddress: 0x200158
# CHECK-NEXT:   PhysicalAddress: 0x200158
# CHECK-NEXT:   FileSize: 8
# CHECK-NEXT:   MemSize: 8
# CHECK-NEXT:   Flags [ (0x4)
# CHECK-NEXT:     PF_R (0x4)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Alignment: 1
# CHECK-NEXT: }

.section .openbsd.randomdata, "a"
.quad 0
