# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --sections -n %t | FileCheck %s

## Check that .note.gnu.property has alignment 8 and is readable by llvm-readobj

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

# CHECK:      Size: 0x20
# CHECK-NEXT: Notes [
# CHECK-NEXT: {
# CHECK-NEXT:   Owner: GNU
# CHECK-NEXT:   Data size: 0x10
# CHECK-NEXT:   Type: NT_GNU_PROPERTY_TYPE_0 (property note)
# CHECK-NEXT:   Property [
# CHECK-NEXT:     x86 feature: IBT


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
 .type _start, %function
_start: ret
