# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_386_TLS_LE against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_LE_32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_LDO_32 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_IE against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_GOTDESC against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_GD against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_386_TLS_LDM against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_386_TLS_LE, nonTls
 nop
 .reloc ., R_386_TLS_LE_32, nonTls
 nop
 .reloc ., R_386_TLS_LDO_32, nonTls
 nop
 .reloc ., R_386_TLS_IE, nonTls
 nop
 .reloc ., R_386_TLS_GOTDESC, nonTls
 nop
## On error, handleTlsGd/handleTlsLd skip the next relocation (normally the
## __tls_get_addr call). Add a benign one to keep the ++it in bounds.
 .reloc ., R_386_TLS_GD, nonTls
 nop
 .reloc ., R_386_NONE
 nop
 .reloc ., R_386_TLS_LDM, nonTls
 nop
 .reloc ., R_386_NONE
 nop

.data
.globl nonTls
nonTls:
 .word 0
