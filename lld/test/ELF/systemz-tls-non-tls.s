# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-linux-gnu %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_390_TLS_GOTIE12 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_390_TLS_LDO32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_390_TLS_IEENT against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_390_TLS_LE32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_390_TLS_GD32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_390_TLS_LDM32 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_390_TLS_GOTIE12, nonTls
 nop
 .reloc ., R_390_TLS_LDO32, nonTls
 nop
 .reloc ., R_390_TLS_IEENT, nonTls
 nop
 .reloc ., R_390_TLS_LE32, nonTls
 nop
 .reloc ., R_390_TLS_GD32, nonTls
 nop
 .reloc ., R_390_TLS_LDM32, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
