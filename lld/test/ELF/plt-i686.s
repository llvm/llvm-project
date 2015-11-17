// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x11010
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.dyn {
// CHECK-NEXT:     0x12050 R_386_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x12054 R_386_GLOB_DAT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:

// 0x11010 - (0x11000 + 1) - 4 = 11
// 0x11010 - (0x11005 + 1) - 4 = 2
// 0x11018 - (0x1100a + 1) - 4 = 9

// DISASM:      _start:
// DISASM-NEXT:   11000:  e9 0b 00 00 00  jmp  11
// DISASM-NEXT:   11005:  e9 06 00 00 00  jmp  6
// DISASM-NEXT:   1100a:  e9 09 00 00 00  jmp  9

// 0x12050 = 73808
// 0x12054 = 73812

// DISASM:      Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   11010:  ff 25 {{.*}}       jmpl *73808
// DISASM-NEXT:   11016:  90                 nop
// DISASM-NEXT:   11017:  90                 nop
// DISASM-NEXT:   11018:  ff 25 {{.*}}       jmpl *73812
// DISASM-NEXT:   1101e:  90                 nop
// DISASM-NEXT:   1101f:  90                 nop

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
