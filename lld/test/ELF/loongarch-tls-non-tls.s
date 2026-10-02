# REQUIRES: loongarch
# RUN: llvm-mc -filetype=obj -triple=loongarch64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_LARCH_TLS_LE_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_GD_PC_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_DESC_PC_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_DTPREL32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_IE_PC_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_LD_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_GD_HI20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_LD_PCREL20_S2 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_GD_PCREL20_S2 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_DESC_CALL against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_DESC64_PC_LO20 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_LARCH_TLS_DESC_HI20 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_LARCH_TLS_LE_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_GD_PC_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_DESC_PC_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_DTPREL32, nonTls
 nop
 .reloc ., R_LARCH_TLS_IE_PC_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_LD_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_GD_HI20, nonTls
 nop
 .reloc ., R_LARCH_TLS_LD_PCREL20_S2, nonTls
 nop
 .reloc ., R_LARCH_TLS_GD_PCREL20_S2, nonTls
 nop
 .reloc ., R_LARCH_TLS_DESC_CALL, nonTls
 nop
 .reloc ., R_LARCH_TLS_DESC64_PC_LO20, nonTls
 nop
 .reloc ., R_LARCH_TLS_DESC_HI20, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
