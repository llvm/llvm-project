// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-got.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so
// RUN: ld.lld -e main %t1.o %t2.so -o %t3
// RUN: llvm-readobj -S -r %t3 | FileCheck %s
// RUN: llvm-objdump -d %t3 | FileCheck --check-prefix=DISASM %s

// CHECK:      Section {
// CHECK:      Index: 9
// CHECK-NEXT: Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[ADDR:.*]]
// CHECK-NEXT: Offset: 0x3B0
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: }

// CHECK:      Relocations [
// CHECK-NEXT:   Section (5) .rela.dyn {
// CHECK-NEXT:     [[ADDR]] R_X86_64_TPOFF64 tls1 0x0
// CHECK-NEXT:     0x2023B8 R_X86_64_TPOFF64 tls0 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// 0x2012d0 + 4313 + 7 = 0x2023B0
// 0x2012dA + 4311 + 7 = 0x2023B8
// 0x2012e4 + 4301 + 7 = 0x2023B8
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <main>:
// DISASM-NEXT: 2012d0: {{.*}} movq 4313(%rip), %rax
// DISASM-NEXT: 2012d7: {{.*}} movl %fs:(%rax), %eax
// DISASM-NEXT: 2012da: {{.*}} movq 4311(%rip), %rax
// DISASM-NEXT: 2012e1: {{.*}} movl %fs:(%rax), %eax
// DISASM-NEXT: 2012e4: {{.*}} movq 4301(%rip), %rax
// DISASM-NEXT: 2012eb: {{.*}} movl %fs:(%rax), %eax
// DISASM-NEXT: 2012ee: {{.*}} retq

.section .tdata,"awT",@progbits

.text
 .globl main
 .align 16, 0x90
 .type main,@function
main:
 movq tls1@GOTTPOFF(%rip), %rax
 movl %fs:0(%rax), %eax
 movq tls0@GOTTPOFF(%rip), %rax
 movl %fs:0(%rax), %eax
 movq tls0@GOTTPOFF(%rip), %rax
 movl %fs:0(%rax), %eax
 ret
