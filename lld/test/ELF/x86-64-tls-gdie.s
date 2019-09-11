// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-opt-gdie.s -o %tso.o
// RUN: ld.lld -shared %tso.o -soname=so -o %t.so
// RUN: ld.lld %t.o %t.so -o %t1
// RUN: llvm-readobj -S %t1 | FileCheck --check-prefix=SEC --implicit-check-not=.plt %s
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOC %s
// RUN: llvm-objdump -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

// SEC .got PROGBITS 00000000002020b0 0020b0 000010 00 WA 0 0 8

//RELOC:      Relocations [
//RELOC-NEXT:   Section (5) .rela.dyn {
//RELOC-NEXT:     0x2020C0 R_X86_64_TPOFF64 tlsshared0 0x0
//RELOC-NEXT:     0x2020C8 R_X86_64_TPOFF64 tlsshared1 0x0
//RELOC-NEXT:   }
//RELOC-NEXT: ]

// 0x2020C0 - (201009+7) = 4272
// 0x2020C8 - (201019+7) = 4264
// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: _start:
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT: 201009:       addq 4272(%rip), %rax
// DISASM-NEXT:               movq %fs:0, %rax
// DISASM-NEXT: 201019:       addq 4264(%rip), %rax

.section .text
.globl _start
_start:
 .byte 0x66
 leaq tlsshared0@tlsgd(%rip),%rdi
 .word 0x6666
 rex64
 call __tls_get_addr@plt
 .byte 0x66
 leaq tlsshared1@tlsgd(%rip),%rdi
 .word 0x6666
 rex64
 call __tls_get_addr@plt
