// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t
// RUN: ld.lld %t -o %tout
// RUN: ld.lld %t -shared -o %tsharedout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DIS
// RUN: llvm-readobj -r %tout | FileCheck %s --check-prefix=RELOC
// RUN: llvm-objdump -d %tsharedout | FileCheck %s --check-prefix=DISSHARED
// RUN: llvm-readobj -r %tsharedout | FileCheck %s --check-prefix=RELOCSHARED

.section ".tdata", "awT", @progbits
.globl var
.globl var1
var:
.long 0
var1:
.long 1

.section test, "awx"
.global _start
_start:
 movl $var@tpoff, %edx
 movl %gs:0, %ecx
 subl %edx, %eax
 movl $var1@tpoff, %edx
 movl %gs:0, %ecx
 subl %edx, %eax

 movl %gs:0, %ecx
 leal var@ntpoff(%ecx), %eax
 movl %gs:0, %ecx
 leal var1@ntpoff+123(%ecx), %eax

// DIS:      Disassembly of section test:
// DIS-NEXT: _start:
// DIS-NEXT: 12000: ba 08 00 00 00       movl $8, %edx
// DIS-NEXT: 12005: 65 8b 0d 00 00 00 00 movl %gs:0, %ecx
// DIS-NEXT: 1200c: 29 d0                subl %edx, %eax
// DIS-NEXT: 1200e: ba 04 00 00 00       movl $4, %edx
// DIS-NEXT: 12013: 65 8b 0d 00 00 00 00 movl %gs:0, %ecx
// DIS-NEXT: 1201a: 29 d0                subl %edx, %eax
// DIS-NEXT: 1201c: 65 8b 0d 00 00 00 00 movl %gs:0, %ecx
// DIS-NEXT: 12023: 8d 81 f8 ff ff ff    leal -8(%ecx), %eax
// DIS-NEXT: 12029: 65 8b 0d 00 00 00 00 movl %gs:0, %ecx
// DIS-NEXT: 12030: 8d 81 77 00 00 00    leal 119(%ecx), %eax

// RELOC: Relocations [
// RELOC-NEXT: ]

// DISSHARED: Disassembly of section test:
// DISSHARED-NEXT: _start:
// DISSHARED-NEXT: 2000: ba 00 00 00 00 movl   $0, %edx
// DISSHARED-NEXT: 2005: 65 8b 0d 00 00 00 00  movl %gs:0, %ecx
// DISSHARED-NEXT: 200c: 29 d0 subl            %edx, %eax
// DISSHARED-NEXT: 200e: ba 00 00 00 00        movl $0, %edx
// DISSHARED-NEXT: 2013: 65 8b 0d 00 00 00 00  movl %gs:0, %ecx
// DISSHARED-NEXT: 201a: 29 d0 subl            %edx, %eax
// DISSHARED-NEXT: 201c: 65 8b 0d 00 00 00 00  movl %gs:0, %ecx
// DISSHARED-NEXT: 2023: 8d 81 00 00 00 00     leal (%ecx), %eax
// DISSHARED-NEXT: 2029: 65 8b 0d 00 00 00 00  movl %gs:0, %ecx
// DISSHARED-NEXT: 2030: 8d 81 7b 00 00 00     leal 123(%ecx), %eax

// RELOCSHARED:      Relocations [
// RELOCSHARED-NEXT: Section (4) .rel.dyn {
// RELOCSHARED-NEXT:   0x2001 R_386_TLS_TPOFF32 var 0x0
// RELOCSHARED-NEXT:   0x2025 R_386_TLS_TPOFF var 0x0
// RELOCSHARED-NEXT:   0x200F R_386_TLS_TPOFF32 var1 0x0
// RELOCSHARED-NEXT:   0x2032 R_386_TLS_TPOFF var1 0x0
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT: ]
