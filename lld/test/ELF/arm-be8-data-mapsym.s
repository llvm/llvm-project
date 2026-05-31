/// REQUIRES: arm
/// ARM big-endian(BE-8) reversal needs to account for explicit mapping
/// symbols in data sections not represented as an InputSection. SHF_MERGE sections
/// are an example of such a section.

// RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld --be8 %t.o -shared -o %t
// RUN: llvm-readelf -S -x .merge %t | FileCheck %s

// CHECK: Hex dump of section '.merge':
// CHECK-NEXT: 0x0000014d 11223344

.section .merge, "aM", %progbits, 4
/// GNU assembler adds a mapping symbol for this SHF_MERGE data section.
/// Clang integrated assembler uses the implicit $d from the section type,
/// so add one manually to match GNU assembler output.
.local $d.1
$d.1:
        .word 0x11223344
