// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-readobj -sections -section-data %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section        foobar,"",@progbits,unique,1
.section        foobar,"T",@progbits,unique,2
.section        foobar,"",@nobits,unique,3
.section        foobar,"",@nobits,unique,4

.section bar, "a"

// Both sections are in the output and that the alloc section is first:
// CHECK:      Name: bar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:  SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x200120

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_TLS
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK:      Name: foobar
// CHECK-NEXT: Type: SHT_NOBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x0

// CHECK-NOT:  Name: foobar

// Test that the string "bar" is merged into "foobar".

// CHECK:       Section {
// CHECK:       Index:
// CHECK:       Name: .shstrtab
// CHECK-NEXT:  Type: SHT_STRTAB
// CHECK-NEXT:  Flags [
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: 0x0
// CHECK-NEXT:  Offset:
// CHECK-NEXT:  Size:
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 1
// CHECK-NEXT:  EntrySize: 0
// CHECK-NEXT:  SectionData (
// CHECK-NEXT:    0000: 00626172 002E7465 78740066 6F6F6261  |.bar..text.fooba|
// CHECK-NEXT:    0010: 72002E63 6F6D6D65 6E74002E 73796D74  |r..comment..symt|
// CHECK-NEXT:    0020: 6162002E 73687374 72746162 002E7374  |ab..shstrtab..st|
// CHECK-NEXT:    0030: 72746162 00                          |rtab.|
// CHECK-NEXT:  )
// CHECK-NEXT:}
// CHECK:        Name: .strtab
// CHECK-NEXT:   Type: SHT_STRTAB (0x3)
// CHECK-NEXT:   Flags [ (0x0)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size: 15
// CHECK-NEXT:   Link: 0
// CHECK-NEXT:   Info: 0
// CHECK-NEXT:   AddressAlignment: 1
// CHECK-NEXT:   EntrySize: 0
// CHECK-NEXT:   SectionData (
// CHECK-NEXT:     0000: 00666F6F 62617200 5F737461 727400 |.foobar._start.|
// CHECK-NEXT:   )
// CHECK-NEXT: }
