// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

// NORELOC:      Relocations [
// NORELOC-NEXT: ]

// DISASM:      <_start>:
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

// LD to LE:
// DISASM-NEXT:   movq %fs:0, %rax
// DISASM-NEXT:   leaq -8(%rax), %rcx
// DISASM-NEXT:   movq %fs:0, %rax
// DISASM-NEXT:   leaq -4(%rax), %rcx

// GD to LE:
// DISASM-NEXT:   movq %fs:0, %rax
// DISASM-NEXT:   leaq -8(%rax), %rax
// DISASM-NEXT:   movq %fs:0, %rax
// DISASM-NEXT:   leaq -4(%rax), %rax

// LD to LE:
// DISASM:     <_DTPOFF64_1>:
// DISASM-NEXT:   clc
// DISASM:      <_DTPOFF64_2>:
// DISASM-NEXT:   cld

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

 // LD to LE
 leaq tls0@tlsld(%rip), %rdi
 callq __tls_get_addr@PLT
 leaq tls0@dtpoff(%rax),%rcx
 leaq tls1@tlsld(%rip), %rdi
 callq __tls_get_addr@PLT
 leaq tls1@dtpoff(%rax),%rcx

 // GD to LE
 .byte 0x66
 leaq tls0@tlsgd(%rip),%rdi
 .word 0x6666
 rex64
 call __tls_get_addr@plt
 .byte 0x66
 leaq tls1@tlsgd(%rip),%rdi
 .word 0x6666
 rex64
 call __tls_get_addr@plt

 // LD to LE
_DTPOFF64_1:
 .quad tls0@DTPOFF
 nop

_DTPOFF64_2:
 .quad tls1@DTPOFF
 nop
