// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-got.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld -e main %t1.o %t2.so -o %t3
// RUN: llvm-readobj -s -r %t3 | FileCheck %s
// RUN: llvm-objdump -d %t3 | FileCheck --check-prefix=DISASM %s

// CHECK:      Section {
// CHECK:      Index: 8
// CHECK-NEXT: Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[ADDR:.*]]
// CHECK-NEXT: Offset: 0x20A0
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: }

// CHECK:      Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     [[ADDR]] R_X86_64_TPOFF64 tls1 0x0
// CHECK-NEXT:     0x120A8 R_X86_64_TPOFF64 tls0 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

//0x11000 + 4249 + 7 = 0x120A0
//0x1100A + 4247 + 7 = 0x120A8
//0x11014 + 4237 + 7 = 0x120A8
//DISASM:      Disassembly of section .text:
//DISASM-NEXT: main:
//DISASM-NEXT: 11000: 48 8b 05 99 10 00 00 movq 4249(%rip), %rax
//DISASM-NEXT: 11007: 64 8b 00 movl %fs:(%rax), %eax
//DISASM-NEXT: 1100a: 48 8b 05 97 10 00 00 movq 4247(%rip), %rax
//DISASM-NEXT: 11011: 64 8b 00 movl %fs:(%rax), %eax
//DISASM-NEXT: 11014: 48 8b 05 8d 10 00 00 movq 4237(%rip), %rax
//DISASM-NEXT: 1101b: 64 8b 00 movl %fs:(%rax), %eax
//DISASM-NEXT: 1101e: c3 retq

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
