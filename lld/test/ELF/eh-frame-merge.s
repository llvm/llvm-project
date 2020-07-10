// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.o %t.o -o %t -shared
// RUN: llvm-readobj -S --section-data %t | FileCheck %s

/// Also show that the merging happens when going via a -r link.
// RUN: ld.lld -r %t.o %t.o -o %t.r.o
// RUN: ld.lld --hash-style=sysv %t.r.o -o %t2 -shared
// RUN: llvm-readobj -S --section-data %t2 | FileCheck %s

        .section	foo,"ax",@progbits
	.cfi_startproc
        nop
	.cfi_endproc

        .section	bar,"axG",@progbits,foo,comdat
        .cfi_startproc
        nop
        nop
	.cfi_endproc

// FIXME: We could really use a .eh_frame parser.
// The intention is to show that:
// * There is only one copy of the CIE
// * There are two copies of the first FDE
// * There is only one copy of the second FDE

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address:
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 100
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT: 0000: 14000000 00000000 017A5200 01781001  |
// CHECK-NEXT: 0010: 1B0C0708 90010000 14000000 1C000000  |
// CHECK-NEXT: 0020: 44100000 01000000 00000000 00000000  |
// CHECK-NEXT: 0030: 14000000 34000000 2E100000 02000000  |
// CHECK-NEXT: 0040: 00000000 00000000 14000000 4C000000  |
// CHECK-NEXT: 0050: 15100000 01000000 00000000 00000000  |
// CHECK-NEXT: 0060: 00000000
// CHECK-NEXT: )

// CHECK:      Name: foo
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x125C

// CHECK:      Name: bar
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x125E
