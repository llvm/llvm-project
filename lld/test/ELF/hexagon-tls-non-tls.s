# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_HEX_GD_PLT_B22_PCREL against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_HEX_GD_GOT_32_6_X against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_HEX_DTPREL_32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_HEX_TPREL_16_X against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_HEX_IE_GOT_11_X against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_HEX_GD_PLT_B22_PCREL, nonTls
 nop
 .reloc ., R_HEX_GD_GOT_32_6_X, nonTls
 nop
 .reloc ., R_HEX_DTPREL_32, nonTls
 nop
 .reloc ., R_HEX_TPREL_16_X, nonTls
 nop
 .reloc ., R_HEX_IE_GOT_11_X, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
