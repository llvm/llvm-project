// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t %t2.so -o %t2
// RUN: llvm-readobj -s %t2 | FileCheck --check-prefix=ADDR %s
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86

.global _start
_start:

.section       .R_386_32,"ax",@progbits
.global R_386_32
R_386_32:
  movl $R_386_32 + 1, %edx


.section       .R_386_PC32,"ax",@progbits,unique,1
.global R_386_PC32
R_386_PC32:
  call R_386_PC32_2

.section       .R_386_PC32,"ax",@progbits,unique,2
.zero 4
R_386_PC32_2:
  nop

// CHECK: Disassembly of section .R_386_32:
// CHECK-NEXT: R_386_32:
// CHECK-NEXT:  11000: {{.*}} movl $69633, %edx

// CHECK: Disassembly of section .R_386_PC32:
// CHECK-NEXT: R_386_PC32:
// CHECK-NEXT:   11005:  e8 04 00 00 00  calll 4

// CHECK:      R_386_PC32_2:
// CHECK-NEXT:   1100e:  90  nop

// Create a .got
movl bar@GOT, %eax

// ADDR:      Name: .plt
// ADDR-NEXT: Type: SHT_PROGBITS
// ADDR-NEXT: Flags [
// ADDR-NEXT:   SHF_ALLOC
// ADDR-NEXT:   SHF_EXECINSTR
// ADDR-NEXT: ]
// ADDR-NEXT: Address: 0x11030
// ADDR-NEXT: Offset: 0x1030
// ADDR-NEXT: Size: 32

// ADDR:      Name: .got
// ADDR-NEXT: Type: SHT_PROGBITS
// ADDR-NEXT: Flags [
// ADDR-NEXT:   SHF_ALLOC
// ADDR-NEXT:   SHF_WRITE
// ADDR-NEXT: ]
// ADDR-NEXT: Address: 0x12078

.section .R_386_GOTPC,"ax",@progbits
R_386_GOTPC:
 movl $_GLOBAL_OFFSET_TABLE_, %eax

// 0x12050 - 0x11014 = 4156

// CHECK:      Disassembly of section .R_386_GOTPC:
// CHECK-NEXT: R_386_GOTPC:
// CHECK-NEXT:   11014:  {{.*}} movl  $4196, %eax

.section .dynamic_reloc, "ax",@progbits
 call bar
// 0x11030 - (0x11019 + 5) = 18
// CHECK:      Disassembly of section .dynamic_reloc:
// CHECK-NEXT: .dynamic_reloc:
// CHECK-NEXT:   11019:  e8 22 00 00 00 calll 34

.section .R_386_GOT32,"ax",@progbits
.global R_386_GOT32
R_386_GOT32:
        movl zed@GOT, %eax
// This is the second symbol in the got, so the offset is 4.
// CHECK:      Disassembly of section .R_386_GOT32:
// CHECK-NEXT: R_386_GOT32:
// CHECK-NEXT:   1101e:  {{.*}} movl 4, %eax
