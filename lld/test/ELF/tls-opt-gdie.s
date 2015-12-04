// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/tls-opt-gdie.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %t.so
// RUN: ld.lld %t.o %t.so -o %t1
// RUN: llvm-readobj -s -r %t1 | FileCheck --check-prefix=RELOC %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

//RELOC:      Section {
//RELOC:      Index: 9
//RELOC-NEXT: Name: .got
//RELOC-NEXT: Type: SHT_PROGBITS
//RELOC-NEXT: Flags [
//RELOC-NEXT:   SHF_ALLOC
//RELOC-NEXT:   SHF_WRITE
//RELOC-NEXT: ]
//RELOC-NEXT: Address: 0x120E0
//RELOC-NEXT: Offset: 0x20E0
//RELOC-NEXT: Size: 16
//RELOC-NEXT: Link: 0
//RELOC-NEXT: Info: 0
//RELOC-NEXT: AddressAlignment: 8
//RELOC-NEXT: EntrySize: 0
//RELOC-NEXT: }
//RELOC:      Relocations [
//RELOC-NEXT:   Section (4) .rela.dyn {
//RELOC-NEXT:     0x120E0 R_X86_64_TPOFF64 tlsshared0 0x0
//RELOC-NEXT:     0x120E8 R_X86_64_TPOFF64 tlsshared1 0x0
//RELOC-NEXT:   }
//RELOC-NEXT:   Section (5) .rela.plt {
//RELOC-NEXT:     0x12108 R_X86_64_JUMP_SLOT __tls_get_addr 0x0
//RELOC-NEXT:   }
//RELOC-NEXT: ]

//0x11009 + (4304 + 7) = 0x120E0
//0x11019 + (4296 + 7) = 0x120E8
// DISASM:      Disassembly of section .text:
// DISASM-NEXT: _start:
// DISASM-NEXT: 11000: 64 48 8b 04 25 00 00 00 00 movq %fs:0, %rax
// DISASM-NEXT: 11009: 48 03 05 d0 10 00 00       addq 4304(%rip), %rax
// DISASM-NEXT: 11010: 64 48 8b 04 25 00 00 00 00 movq %fs:0, %rax
// DISASM-NEXT: 11019: 48 03 05 c8 10 00 00       addq 4296(%rip), %rax

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
