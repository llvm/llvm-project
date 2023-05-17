// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686 %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld --hash-style=sysv %t.o %t2.so -o %t
// RUN: llvm-readobj -S %t | FileCheck --check-prefix=ADDR %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

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
// CHECK-EMPTY:
// CHECK-NEXT: <R_386_32>:
// CHECK-NEXT:   movl $4198829, %edx

// CHECK: Disassembly of section .R_386_PC32:
// CHECK-EMPTY:
// CHECK-NEXT: <R_386_PC32>:
// CHECK-NEXT:   calll 0x4011ba

// CHECK:      <R_386_PC32_2>:
// CHECK-NEXT:   nop

// Create a .got
movl bar@GOT, %eax

// ADDR:      Name: .plt
// ADDR-NEXT: Type: SHT_PROGBITS
// ADDR-NEXT: Flags [
// ADDR-NEXT:   SHF_ALLOC
// ADDR-NEXT:   SHF_EXECINSTR
// ADDR-NEXT: ]
// ADDR-NEXT: Address: 0x4011E0
// ADDR-NEXT: Offset: 0x1E0
// ADDR-NEXT: Size: 32

// ADDR:      Name: .got.plt (
// ADDR-NEXT: Type: SHT_PROGBITS
// ADDR-NEXT: Flags [
// ADDR-NEXT:   SHF_ALLOC
// ADDR-NEXT:   SHF_WRITE
// ADDR-NEXT: ]
// ADDR-NEXT: Address: 0x403280
// ADDR-NEXT: Offset:
// ADDR-NEXT: Size:

.section .R_386_GOTPC,"ax",@progbits
R_386_GOTPC:
 movl $_GLOBAL_OFFSET_TABLE_, %eax

// .got.plt - 0x4011c0 = 0x403280 - 0x4011c0 = 8384
// CHECK:      Disassembly of section .R_386_GOTPC:
// CHECK-EMPTY:
// CHECK-NEXT: <R_386_GOTPC>:
// CHECK-NEXT:   4011c0:       movl  $8384, %eax

.section .dynamic_reloc, "ax",@progbits
 call bar
// .plt + 16 - (0x4011c5 + 5) = 0x4011e0 + 16 - 0x4011ca = 38
// CHECK:      Disassembly of section .dynamic_reloc:
// CHECK-EMPTY:
// CHECK-NEXT: <.dynamic_reloc>:
// CHECK-NEXT:   4011c5:       calll 0x4011f0 <bar@plt>

.section .R_386_GOT32,"ax",@progbits
.global R_386_GOT32
R_386_GOT32:
 movl bar@GOT, %eax
 movl zed@GOT, %eax
 movl bar+8@GOT, %eax
 movl zed+4@GOT, %eax

// &.got[0] - .got.plt = 0x402278 - 0x403280 = 4294963192
// &.got[1] - .got.plt = 0x402278 + 4 - 0x403280 = 4294963196
// &.got[2] - .got.plt = 0x402278 + 8 - 0x403280 = 4294963200
// CHECK:      Disassembly of section .R_386_GOT32:
// CHECK-EMPTY:
// CHECK-NEXT: <R_386_GOT32>:
// CHECK-NEXT: 4011ca:       movl 4294963192, %eax
// CHECK-NEXT:               movl 4294963196, %eax
// CHECK-NEXT:               movl 4294963200, %eax
// CHECK-NEXT:               movl 4294963200, %eax
