# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %p/Inputs/aarch64-tls-ie.s -o %ttlsie.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %tmain.o
# RUN: ld.lld %tmain.o %ttlsie.o -o %tout
# RUN: llvm-objdump -d %tout | FileCheck %s
# RUN: llvm-readobj -s -r %tout | FileCheck -check-prefix=RELOC %s
# REQUIRES: aarch64

#Local-Dynamic to Initial-Exec relax creates no
#RELOC:      Relocations [
#RELOC-NEXT: ]

# TCB size = 0x16 and foo is first element from TLS register.
#CHECK: Disassembly of section .text:
#CHECK: _start:
#CHECK:  11000:	00 00 a0 d2	movz	x0, #0, lsl #16
#CHECK:  11004:	00 02 80 f2 	movk	x0, #0x10

.globl _start
_start:
 adrp    x0, :gottprel:foo
 ldr     x0, [x0, :gottprel_lo12:foo]
