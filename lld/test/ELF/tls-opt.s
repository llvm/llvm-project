// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

// NORELOC:      Relocations [
// NORELOC-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-NEXT: _start:
// DISASM-NEXT: 11000: 48 c7 c0 f8 ff ff ff movq $-8, %rax
// DISASM-NEXT: 11007: 49 c7 c7 f8 ff ff ff movq $-8, %r15
// DISASM-NEXT: 1100e: 48 8d 80 f8 ff ff ff leaq -8(%rax), %rax
// DISASM-NEXT: 11015: 4d 8d bf f8 ff ff ff leaq -8(%r15), %r15
// DISASM-NEXT: 1101c: 48 81 c4 f8 ff ff ff addq $-8, %rsp
// DISASM-NEXT: 11023: 49 81 c4 f8 ff ff ff addq $-8, %r12
// DISASM-NEXT: 1102a: 48 c7 c0 fc ff ff ff movq $-4, %rax
// DISASM-NEXT: 11031: 49 c7 c7 fc ff ff ff movq $-4, %r15
// DISASM-NEXT: 11038: 48 8d 80 fc ff ff ff leaq -4(%rax), %rax
// DISASM-NEXT: 1103f: 4d 8d bf fc ff ff ff leaq -4(%r15), %r15
// DISASM-NEXT: 11046: 48 81 c4 fc ff ff ff addq $-4, %rsp
// DISASM-NEXT: 1104d: 49 81 c4 fc ff ff ff addq $-4, %r12

// Corrupred output:
// DISASM-NEXT: 11054: 48 8d 80 f8 ff ff ff leaq -8(%rax), %rax
// DISASM-NEXT: 1105b: 48 d1 81 c4 f8 ff ff rolq -1852(%rcx)
// DISASM-NEXT: 11062: ff 48 d1 decl -47(%rax)
// DISASM-NEXT: 11065: 81 c4 f8 ff ff ff addl $4294967288, %esp

.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
.align 4
tls0:
 .long 0
 .size tls0, 4

.type  tls1,@object
.globl tls1
.align 4
tls1:
 .long 0
 .size tls1, 4

.section .text
.globl _start
_start:
 movq tls0@GOTTPOFF(%rip), %rax
 movq tls0@GOTTPOFF(%rip), %r15
 addq tls0@GOTTPOFF(%rip), %rax
 addq tls0@GOTTPOFF(%rip), %r15
 addq tls0@GOTTPOFF(%rip), %rsp
 addq tls0@GOTTPOFF(%rip), %r12
 movq tls1@GOTTPOFF(%rip), %rax
 movq tls1@GOTTPOFF(%rip), %r15
 addq tls1@GOTTPOFF(%rip), %rax
 addq tls1@GOTTPOFF(%rip), %r15
 addq tls1@GOTTPOFF(%rip), %rsp
 addq tls1@GOTTPOFF(%rip), %r12

 //Invalid input case:
 xchgq tls0@gottpoff(%rip),%rax
 shlq tls0@gottpoff
 rolq tls0@gottpoff
