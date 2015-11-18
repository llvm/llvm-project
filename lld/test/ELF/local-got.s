// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r -section-data %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

        .globl _start
_start:
	call bar@gotpcrel
	call foo@gotpcrel

        .global foo
foo:
        nop

// 0x120A0 - 0x11000 - 5 =  4251
// 0x120A8 - 0x11005 - 5 =  4254
// DISASM:      _start:
// DISASM-NEXT:   11000: {{.*}} callq 4251
// DISASM-NEXT:   11005: {{.*}} callq 4254

// DISASM:      foo:
// DISASM-NEXT:   1100a: {{.*}} nop

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x120A0
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// 0x1200a in little endian
// CHECK-NEXT:   0000:  00000000 00000000 0A100100 00000000
// CHECK-NEXT: )

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x120A0 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
