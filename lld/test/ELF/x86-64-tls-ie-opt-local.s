// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

// NORELOC:      Relocations [
// NORELOC-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: _start:
// DISASM-NEXT:   movq $-8, %rax
// DISASM-NEXT:   movq $-8, %r15
// DISASM-NEXT:   leaq -8(%rax), %rax
// DISASM-NEXT:   leaq -8(%r15), %r15
// DISASM-NEXT:   addq $-8, %rsp
// DISASM-NEXT:   addq $-8, %r12
// DISASM-NEXT:   movq $-4, %rax
// DISASM-NEXT:   movq $-4, %r15
// DISASM-NEXT:   leaq -4(%rax), %rax
// DISASM-NEXT:   leaq -4(%r15), %r15
// DISASM-NEXT:   addq $-4, %rsp
// DISASM-NEXT:   addq $-4, %r12

.section .tbss,"awT",@nobits

.type tls0,@object
.align 4
tls0:
 .long 0
 .size tls0, 4

.type  tls1,@object
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
