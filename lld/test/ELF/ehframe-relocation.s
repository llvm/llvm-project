// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: echo '.cfi_startproc; .cfi_endproc' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t
// RUN: llvm-readobj -S %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x200120
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 52
// CHECK-NOT: .eh_frame

// 0x200120 = 2097440
// 0x200120 + 5 = 2097445
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: _start:
// DISASM-NEXT:   201154: {{.*}} movq 2097440, %rax
// DISASM-NEXT:   20115c: {{.*}} movq 2097445, %rax

.section .eh_frame,"ax",@unwind

.section .text
.globl _start
_start:
 movq .eh_frame, %rax
 movq .eh_frame + 5, %rax
