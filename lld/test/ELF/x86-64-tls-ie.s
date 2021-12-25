// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-got.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so
// RUN: ld.lld -e main %t1.o %t2.so -o %t3
// RUN: llvm-readobj -S -r %t3 | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t3 | FileCheck --check-prefix=DISASM %s

// CHECK:      Section {
// CHECK:      Index: 9
// CHECK-NEXT: Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: [[ADDR:.*]]
// CHECK-NEXT: Offset: 0x3F0
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8
// CHECK-NEXT: EntrySize: 0
// CHECK-NEXT: }

// CHECK:      Relocations [
// CHECK-NEXT:   Section (5) .rela.dyn {
// CHECK-NEXT:     [[ADDR]] R_X86_64_TPOFF64 tls1 0x0
// CHECK-NEXT:     0x2023F8 R_X86_64_TPOFF64 tls0 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

/// 0x2023F0 - 0x201307 = 4329
/// 0x2023F8 - 0x201311 = 4327
/// 0x2023F8 - 0x20131b = 4317
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <main>:
// DISASM-NEXT:         movq 4329(%rip), %rax
// DISASM-NEXT: 201307: movl %fs:(%rax), %eax
// DISASM-NEXT:         movq 4327(%rip), %rax
// DISASM-NEXT: 201311: movl %fs:(%rax), %eax
// DISASM-NEXT:         movq 4317(%rip), %rax
// DISASM-NEXT: 20131b: movl %fs:(%rax), %eax

/// 0x2023F0 - 0x20132e = 4290
// DISASM-NEXT:         movq %fs:0, %rax
// DISASM-NEXT:         addq 4290(%rip), %rax
// DISASM-NEXT: 20132e: retq

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

## Relaxed to TLS IE. Share the GOT entry with GOTTPOFF.
 .byte   0x66
 leaq    tls1@tlsgd(%rip), %rdi
 .value  0x6666
 rex64
 call    __tls_get_addr@PLT
 ret
