# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %p/Inputs/aarch64-tls-ie.s -o %ttlsie.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %tmain.o
# RUN: ld.lld %tmain.o %ttlsie.o -o %tout
# RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s
# RUN: llvm-readobj -r %tout | FileCheck -check-prefix=RELOC %s

## Local-Dynamic to Local-Exec relax creates no dynamic relocations.
# RELOC:      Relocations [
# RELOC-NEXT: ]

# TCB size = 0x16 and foo is first element from TLS register.
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    2101c8: movz    x0, #0, lsl #16
# CHECK-NEXT:    2101cc: movk    x0, #16
# CHECK-NEXT:    2101d0: nop
# CHECK-NEXT:    2101d4: nop

.globl _start
_start:
 adrp    x0, :tlsdesc:foo
 ldr     x1, [x0, :tlsdesc_lo12:foo]
 add     x0, x0, :tlsdesc_lo12:foo
 .tlsdesccall foo
 blr     x1
