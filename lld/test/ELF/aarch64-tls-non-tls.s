# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_AARCH64_TLSDESC_ADR_PAGE21 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_AARCH64_AUTH_TLSDESC_ADR_PAGE21 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21, nonTls
 nop
 .reloc ., R_AARCH64_TLSDESC_ADR_PAGE21, nonTls
 nop
 .reloc ., R_AARCH64_AUTH_TLSDESC_ADR_PAGE21, nonTls
 nop

.data
.type nonTls,@object
.globl nonTls
nonTls:
 .word 0
.size nonTls, 4
