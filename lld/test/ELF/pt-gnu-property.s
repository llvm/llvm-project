# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --sections --program-headers %t | FileCheck %s

## Test that we generate the PT_GNU_PROPERTY segment type that describes the
## .note.gnu.property if it is present.

# CHECK:      Name: .note.gnu.property
# CHECK-NEXT: Type: SHT_NOTE (0x7)
# CHECK-NEXT: Flags [ (0x2)
# CHECK-NEXT:   SHF_ALLOC (0x2)
# CHECK-NEXT: ]
# CHECK-NEXT: Address: 0x200190
# CHECK-NEXT: Offset: 0x190
# CHECK-NEXT: Size: 32
# CHECK-NEXT: Link: 0
# CHECK-NEXT: Info: 0
# CHECK-NEXT: AddressAlignment: 8

# CHECK:      Type: PT_GNU_PROPERTY (0x6474E553)
# CHECK-NEXT: Offset: 0x190
# CHECK-NEXT: VirtualAddress: 0x200190
# CHECK-NEXT: PhysicalAddress: 0x200190
# CHECK-NEXT: FileSize: 32
# CHECK-NEXT: MemSize: 32
# CHECK-NEXT: Flags [ (0x4)
# CHECK-NEXT:   PF_R (0x4)
# CHECK-NEXT: ]
# CHECK-NEXT: Alignment: 8

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000002 # GNU_PROPERTY_X86_FEATURE_1_AND
.long 4
.long 1          # GNU_PROPERTY_X86_FEATURE_1_IBT
.long 0

.text
.globl _start
 ret
