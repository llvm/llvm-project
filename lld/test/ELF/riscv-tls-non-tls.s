# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_RISCV_TLS_GD_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_RISCV_TPREL_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_RISCV_TLSDESC_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_RISCV_TLS_GOT_HI20 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_RISCV_TLS_GD_HI20, nonTls
 nop
 .reloc ., R_RISCV_TPREL_HI20, nonTls
 nop
 .reloc ., R_RISCV_TLSDESC_HI20, nonTls
 nop
 .reloc ., R_RISCV_TLS_GOT_HI20, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
