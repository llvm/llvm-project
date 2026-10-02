# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux-gnu %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_PPC64_TLS against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_GOT_TLSGD16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_TLSGD against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_GOT_TLSLD16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_DTPREL16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_GOT_DTPREL16_HA against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_TPREL16 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_PPC64_GOT_TPREL16_DS against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_PPC64_TLS, nonTls
 nop
 .reloc ., R_PPC64_GOT_TLSGD16, nonTls
 nop
 .reloc ., R_PPC64_TLSGD, nonTls
 nop
## R_PPC64_TLSGD consumes the next relocation (normally the __tls_get_addr
## call) via ++it; add a benign one so the following relocations are scanned.
 .reloc ., R_PPC64_REL24
 nop
 .reloc ., R_PPC64_GOT_TLSLD16, nonTls
 nop
 .reloc ., R_PPC64_DTPREL16, nonTls
 nop
 .reloc ., R_PPC64_GOT_DTPREL16_HA, nonTls
 nop
 .reloc ., R_PPC64_TPREL16, nonTls
 nop
 .reloc ., R_PPC64_GOT_TPREL16_DS, nonTls
 nop

.data
.globl nonTls
nonTls:
 .word 0
