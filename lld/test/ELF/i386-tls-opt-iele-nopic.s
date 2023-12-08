// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/tls-opt-iele-i686-nopic.s -o %tso.o
// RUN: ld.lld -shared %tso.o -soname=t.so -o %tso
// RUN: ld.lld --hash-style=sysv %t.o %tso -o %t1
// RUN: llvm-readobj -S -r %t1 | FileCheck --check-prefix=GOTREL %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t1 | FileCheck --check-prefix=DISASM %s

// GOTREL:      Section {
// GOTREL:        Index:
// GOTREL:        Name: .got
// GOTREL-NEXT:   Type: SHT_PROGBITS
// GOTREL-NEXT:   Flags [
// GOTREL-NEXT:     SHF_ALLOC
// GOTREL-NEXT:     SHF_WRITE
// GOTREL-NEXT:   ]
// GOTREL-NEXT:   Address:  0x402250
// GOTREL-NEXT:   Offset: 0x250
// GOTREL-NEXT:   Size: 8
// GOTREL-NEXT:   Link: 0
// GOTREL-NEXT:   Info: 0
// GOTREL-NEXT:   AddressAlignment: 4
// GOTREL-NEXT:   EntrySize: 0
// GOTREL-NEXT: }
// GOTREL:      Relocations [
// GOTREL-NEXT: Section ({{.*}}) .rel.dyn {
// GOTREL-NEXT:   0x402250 R_386_TLS_TPOFF tlsshared0
// GOTREL-NEXT:   0x402254 R_386_TLS_TPOFF tlsshared1
// GOTREL-NEXT:  }
// GOTREL-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// DISASM-NEXT: 4011b0:       movl $0xfffffff8, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax
// DISASM-NEXT:               movl $0xfffffff8, %eax
// DISASM-NEXT:               movl %gs:(%eax), %eax
// DISASM-NEXT:               addl $0xfffffff8, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax
// DISASM-NEXT:               movl $0xfffffffc, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax
// DISASM-NEXT:               movl $0xfffffffc, %eax
// DISASM-NEXT:               movl %gs:(%eax), %eax
// DISASM-NEXT:               addl $0xfffffffc, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax
/// &.got[0]
// DISASM-NEXT:               movl 0x402250, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax
/// &.got[1]
// DISASM-NEXT:               addl 0x402254, %ecx
// DISASM-NEXT:               movl %gs:(%ecx), %eax

.type tlslocal0,@object
.section .tbss,"awT",@nobits
.globl tlslocal0
.align 4
tlslocal0:
 .long 0
 .size tlslocal0, 4

.type tlslocal1,@object
.section .tbss,"awT",@nobits
.globl tlslocal1
.align 4
tlslocal1:
 .long 0
 .size tlslocal1, 4

.section .text
.globl ___tls_get_addr
.type ___tls_get_addr,@function
___tls_get_addr:

.section .text
.globl _start
_start:
movl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal0@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal1@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlsshared0@indntpoff,%ecx
movl %gs:(%ecx),%eax

addl tlsshared1@indntpoff,%ecx
movl %gs:(%ecx),%eax
