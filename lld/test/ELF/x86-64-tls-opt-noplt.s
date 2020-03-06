// REQUIRES: x86

// Checks whether the TLS optimizations match the cases in Chapter 11 of
// https://raw.githubusercontent.com/wiki/hjl-tools/x86-psABI/x86-64-psABI-1.0.pdf

// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-opt-gdie.s -o %tso.o
// RUN: ld.lld -shared %tso.o -soname=t.so -o %t.so
// RUN: ld.lld %t.o %t.so -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOC %s
// RUN: llvm-objdump -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

// RELOC:      Relocations [
// RELOC-NEXT:  Section {{.*}} .rela.dyn {
// RELOC-NEXT:    0x202420 R_X86_64_TPOFF64 tlsshared0 0x0
// RELOC-NEXT:    0x202428 R_X86_64_TPOFF64 tlsshared1 0x0
// RELOC-NEXT:  }
// RELOC-NEXT: ]

// DISASM:      <_start>:

// Table 11.5: GD -> IE Code Transition (LP64)
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT: 201309:       addq 4368(%rip), %rax
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT: 201319:       addq 4360(%rip), %rax

// Table 11.7: GD -> LE Code Transition (LP64)
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT:               leaq -8(%rax), %rax
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT:               leaq -4(%rax), %rax


// Table 11.9: LD -> LE Code Transition (LP64)
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT:               movq %fs:0, %rax

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
 // Table 11.5: GD -> IE Code Transition (LP64)
 .byte  0x66
 leaq   tlsshared0@tlsgd(%rip),%rdi
 .byte  0x66
 rex64
 call   *__tls_get_addr@GOTPCREL(%rip)

 .byte  0x66
 leaq   tlsshared1@tlsgd(%rip),%rdi
 .byte  0x66
 rex64
 call   *__tls_get_addr@GOTPCREL(%rip)

 // Table 11.7: GD -> LE Code Transition (LP64)
 .byte  0x66
 leaq   tls0@tlsgd(%rip),%rdi
 .byte  0x66
 rex64
 call   *__tls_get_addr@GOTPCREL(%rip)

 .byte  0x66
 leaq   tls1@tlsgd(%rip),%rdi
 .byte  0x66
 rex64
 call   *__tls_get_addr@GOTPCREL(%rip)

 // Table 11.9: LD -> LE Code Transition (LP64)
 leaq   tls0@tlsld(%rip),%rdi
 call   *__tls_get_addr@GOTPCREL(%rip)

 leaq   tls1@tlsld(%rip),%rdi
 call   *__tls_get_addr@GOTPCREL(%rip)
