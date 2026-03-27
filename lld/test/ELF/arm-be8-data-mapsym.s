/// REQUIRES: arm
/// ARM big-endian(BE-8) reversal needs to account for explicit mapping
/// symbols in data sections not represented as an InputSection. SHF_MERGE sections
/// are an example of such a section.

// RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld --be8 %t.o -shared -o %t
// RUN: llvm-readobj --file-headers --sections --section-data %t | FileCheck %s

// CHECK:      DataEncoding: BigEndian (0x2)
// CHECK:      Name: .merge
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_MERGE
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 4
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment:
// CHECK-NEXT: EntrySize: 4
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 11223344 |
// CHECK-NEXT: )

.section .merge, "aM", %progbits, 4
/// GNU assembler adds a mapping symbol for this SHF_MERGE data section.
/// Clang integrated assembler uses the implicit $d from the section type,
/// so add one manually to match GNU assembler output.
.local $d.1
$d.1:
        .word 0x11223344
