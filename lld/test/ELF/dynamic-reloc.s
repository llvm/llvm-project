// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/dynamic-reloc.s -o %t3.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so
// RUN: ld.lld %t.o %t3.o %t2.so -o %t
// RUN: llvm-readobj --dynamic-table -r -S %t | FileCheck %s

// CHECK:      Index: 1
// CHECK-NEXT: Name: .dynsym

// CHECK:      Name: .rela.plt
// CHECK-NEXT: Type: SHT_RELA
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_INFO_LINK
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[RELAADDR:.*]]
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: [[RELASIZE:.*]]
// CHECK-NEXT: Link: 1
// CHECK-NEXT: Info: 9
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 24

// CHECK:      Name: .text
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]

// CHECK: DynamicSection [
// CHECK-NEXT:  Tag                Type                 Name/Value
// CHECK-NEXT:  0x0000000000000001 NEEDED               Shared library: [so]
// CHECK-NEXT:  0x0000000000000015 DEBUG                0x0
// CHECK-NEXT:  0x0000000000000017 JMPREL
// CHECK-NEXT:  0x0000000000000002 PLTRELSZ             24 (bytes)
// CHECK-NEXT:  0x0000000000000003 PLTGOT
// CHECK-NEXT:  0x0000000000000014 PLTREL               RELA
// CHECK-NEXT:  0x0000000000000006 SYMTAB
// CHECK-NEXT:  0x000000000000000B SYMENT               24 (bytes)
// CHECK-NEXT:  0x0000000000000005 STRTAB
// CHECK-NEXT:  0x000000000000000A STRSZ
// CHECK-NEXT:  0x000000006FFFFEF5 GNU_HASH
// CHECK-NEXT:  0x0000000000000004 HASH
// CHECK-NEXT:  0x0000000000000000 NULL                 0x0
// CHECK-NEXT: ]

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     R_X86_64_JUMP_SLOT bar 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.global _start
_start:
.quad bar + 0x42
.weak foo
.quad foo
call main
