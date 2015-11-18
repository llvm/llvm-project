// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -O2 %t.o -o %t.so -shared
// RUN: llvm-readobj -s -section-data -t %t.so | FileCheck %s
// RUN: ld.lld -O1 %t.o -o %t.so -shared
// RUN: llvm-readobj -s -section-data -t %t.so | FileCheck --check-prefix=NOTAIL %s

        .section	.rodata.str1.1,"aMS",@progbits,1
	.asciz	"abc"
foo:
	.ascii	"a"
bar:
        .asciz  "bc"
        .asciz  "bc"

        .section        .rodata.str2.2,"aMS",@progbits,2
        .align  2
zed:
        .short  20
        .short  0

// CHECK:      Name:    .rodata
// CHECK-NEXT: Type:    SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_MERGE
// CHECK-NEXT:   SHF_STRINGS
// CHECK-NEXT: ]
// CHECK-NEXT: Address:         0x190
// CHECK-NEXT: Offset:  0x190
// CHECK-NEXT: Size:    4
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 1
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 61626300                             |abc.|
// CHECK-NEXT: )

// NOTAIL:      Name:    .rodata
// NOTAIL-NEXT: Type:    SHT_PROGBITS
// NOTAIL-NEXT: Flags [
// NOTAIL-NEXT:   SHF_ALLOC
// NOTAIL-NEXT:   SHF_MERGE
// NOTAIL-NEXT:   SHF_STRINGS
// NOTAIL-NEXT: ]
// NOTAIL-NEXT: Address:         0x190
// NOTAIL-NEXT: Offset:  0x190
// NOTAIL-NEXT: Size:    7
// NOTAIL-NEXT: Link: 0
// NOTAIL-NEXT: Info: 0
// NOTAIL-NEXT: AddressAlignment: 1
// NOTAIL-NEXT: EntrySize: 0
// NOTAIL-NEXT: SectionData (
// NOTAIL-NEXT:   0000: 61626300 626300                     |abc.bc.|
// NOTAIL-NEXT: )

// CHECK:      Name: .rodata
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_MERGE
// CHECK-NEXT:   SHF_STRINGS
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x194
// CHECK-NEXT: Offset: 0x194
// CHECK-NEXT: Size: 4
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 2
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000: 14000000                             |....|
// CHECK-NEXT: )


// CHECK:      Name:    bar
// CHECK-NEXT: Value:   0x191

// CHECK:      Name:    foo
// CHECK-NEXT: Value:   0x190

// CHECK:      Name: zed
// CHECK-NEXT: Value: 0x194
// CHECK-NEXT: Size: 0
