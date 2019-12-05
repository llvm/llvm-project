# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --sections -n %t | FileCheck %s

## Check that .note.gnu.property has alignment 4 and is readable by llvm-readobj

# CHECK: Name: .note.gnu.property
# CHECK-NEXT: Type: SHT_NOTE (0x7)
# CHECK-NEXT: Flags [ (0x2)
# CHECK-NEXT:   SHF_ALLOC (0x2)
# CHECK-NEXT: ]
# CHECK-NEXT: Address: 0x4000F4
# CHECK-NEXT: Offset: 0xF4
# CHECK-NEXT: Size: 28
# CHECK-NEXT: Link: 0
# CHECK-NEXT: Info: 0
# CHECK-NEXT: AddressAlignment: 4

# CHECK:      Note {
# CHECK-NEXT:   Owner: GNU
# CHECK-NEXT:   Data size: 0xC
# CHECK-NEXT:   Type: NT_GNU_PROPERTY_TYPE_0 (property note)
# CHECK-NEXT:   Property [
# CHECK-NEXT:     x86 feature: IBT

.section ".note.gnu.property", "a"
.p2align 2
.long 4
.long 0xc
.long 0x5
.asciz "GNU"
.p2align 2
.long 0xc0000002 # GNU_PROPERTY_X86_FEATURE_1_AND
.long 4
.long 1          # GNU_PROPERTY_X86_FEATURE_1_IBT

.text
.globl _start
 ret
