# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7-linux-gnueabi %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_ARM_TLS_LE32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_ARM_TLS_GD32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_ARM_TLS_LDM32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_ARM_TLS_IE32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_ARM_TLS_LDO32 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_ARM_TLS_LE32, nonTls
 nop
 .reloc ., R_ARM_TLS_GD32, nonTls
 nop
 .reloc ., R_ARM_TLS_LDM32, nonTls
 nop
 .reloc ., R_ARM_TLS_IE32, nonTls
 nop
 .reloc ., R_ARM_TLS_LDO32, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
