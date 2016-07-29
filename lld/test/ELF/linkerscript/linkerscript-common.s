# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { .common : { *(COMMON) } }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-readobj -s -t %t1 | FileCheck %s

# q2 alignment is greater than q1, so it should have smaller offset
# because of sorting
# CHECK:       Section {
# CHECK:         Index: 1
# CHECK-NEXT:    Name: .common (1)
# CHECK-NEXT:    Type: SHT_NOBITS (0x8)
# CHECK-NEXT:    Flags [ (0x3)
# CHECK-NEXT:      SHF_ALLOC (0x2)
# CHECK-NEXT:      SHF_WRITE (0x1)
# CHECK-NEXT:    ]
# CHECK-NEXT:    Address: 0x200
# CHECK-NEXT:    Offset: 0x158
# CHECK-NEXT:    Size: 256
# CHECK-NEXT:    Link: 0
# CHECK-NEXT:    Info: 0
# CHECK-NEXT:    AddressAlignment: 256
# CHECK-NEXT:    EntrySize: 0
# CHECK-NEXT:  }
# CHECK:       Symbol {
# CHECK:         Name: q1 (8)
# CHECK-NEXT:    Value: 0x280
# CHECK-NEXT:    Size: 128
# CHECK-NEXT:    Binding: Global (0x1)
# CHECK-NEXT:    Type: Object (0x1)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: .common (0x1)
# CHECK-NEXT:  }
# CHECK-NEXT:  Symbol {
# CHECK-NEXT:    Name: q2 (11)
# CHECK-NEXT:    Value: 0x200
# CHECK-NEXT:    Size: 128
# CHECK-NEXT:    Binding: Global (0x1)
# CHECK-NEXT:    Type: Object (0x1)
# CHECK-NEXT:    Other: 0
# CHECK-NEXT:    Section: .common (0x1)
# CHECK-NEXT:  }

.globl _start
_start:
  jmp _start

.comm q1,128,8
.comm q2,128,256
