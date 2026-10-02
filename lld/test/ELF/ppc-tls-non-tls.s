# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-linux-gnu %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_PPC_TLSGD against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC_DTPREL16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC_TLS against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC_TPREL16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC_GOT_TPREL16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC_GOT_TLSLD16 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_PPC_TLSGD, nonTls
 nop
 .reloc ., R_PPC_DTPREL16, nonTls
 nop
 .reloc ., R_PPC_TLS, nonTls
 nop
 .reloc ., R_PPC_TPREL16, nonTls
 nop
 .reloc ., R_PPC_GOT_TPREL16, nonTls
 nop
 .reloc ., R_PPC_GOT_TLSLD16, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
