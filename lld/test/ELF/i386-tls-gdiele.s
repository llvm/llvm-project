// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686 %p/Inputs/tls-opt-gdiele-i686.s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t1.o
// RUN: ld.lld -shared %t.o -soname=t.so -o %t.so
// RUN: ld.lld --hash-style=sysv %t1.o %t.so -o %tout
// RUN: llvm-readobj -r %tout | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %tout | FileCheck --check-prefix=DISASM %s

// NORELOC:      Relocations [
// NORELOC-NEXT: Section ({{.*}}) .rel.dyn {
// NORELOC-NEXT:   0x402258 R_386_TLS_TPOFF tlsshared0
// NORELOC-NEXT:   0x40225C R_386_TLS_TPOFF tlsshared1
// NORELOC-NEXT:   }
// NORELOC-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// DISASM-NEXT: 4011d0:       movl %gs:0, %eax
// DISASM-NEXT:               addl -4104(%ebx), %eax
// DISASM-NEXT:               movl %gs:0, %eax
// DISASM-NEXT:               addl -4100(%ebx), %eax
// DISASM-NEXT:               movl %gs:0, %eax
// DISASM-NEXT:               subl $8, %eax
// DISASM-NEXT:               movl %gs:0, %eax
// DISASM-NEXT:               subl $4, %eax

.type tlsexe1,@object
.section .tbss,"awT",@nobits
.globl tlsexe1
.align 4
tlsexe1:
 .long 0
 .size tlsexe1, 4

.type tlsexe2,@object
.section .tbss,"awT",@nobits
.globl tlsexe2
.align 4
tlsexe2:
 .long 0
 .size tlsexe2, 4

.section .text
.globl ___tls_get_addr
.type ___tls_get_addr,@function
___tls_get_addr:

.section .text
.globl _start
_start:
//GD->IE
leal tlsshared0@tlsgd(,%ebx,1),%eax
call ___tls_get_addr@plt
leal tlsshared1@tlsgd(,%ebx,1),%eax
call ___tls_get_addr@plt
//GD->LE
leal tlsexe1@tlsgd(,%ebx,1),%eax
call ___tls_get_addr@plt
leal tlsexe2@tlsgd(,%ebx,1),%eax
call ___tls_get_addr@plt
