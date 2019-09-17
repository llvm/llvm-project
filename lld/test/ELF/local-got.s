// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r --section-data %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

        .globl _start
_start:
	call bar@gotpcrel
	call foo@gotpcrel

        .global foo
foo:
        nop

// 0x202320 - 0x201250 - 5 =  4299
// 0x202328 - 0x201255 - 5 =  4302
// DISASM:      _start:
// DISASM-NEXT:   201250:       callq 4299
// DISASM-NEXT:   201255:       callq 4302

// DISASM:      foo:
// DISASM-NEXT:   20125a:       nop

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x202320
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: SectionData (
// CHECK-NEXT:   0000:  00000000 00000000 5A122000 00000000
// CHECK-NEXT: )

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x202320 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]
