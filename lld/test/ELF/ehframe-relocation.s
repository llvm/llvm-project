// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj -s %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_X86_64_UNWIND
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x10120
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 0

// 0x10120 = 65824
// 0x10120 + 5 = 65829
// DISASM:      Disassembly of section .text:
// DISASM-NEXT: _start:
// DISASM-NEXT:    11000: {{.*}} movq 65824, %rax
// DISASM-NEXT:    11008: {{.*}} movq 65829, %rax

.section .eh_frame,"ax",@unwind

.section .text
.globl _start
_start:
 movq .eh_frame, %rax
 movq .eh_frame + 5, %rax
