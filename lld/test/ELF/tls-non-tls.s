# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
# RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# CHECK: error: relocation R_X86_64_TPOFF64 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_X86_64_GOTTPOFF against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_X86_64_TLSGD against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_X86_64_GOTPC32_TLSDESC against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_X86_64_DTPOFF64 against nonTls cannot be used with a non-STT_TLS symbol
# CHECK: error: relocation R_X86_64_TLSLD against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 .reloc ., R_X86_64_TPOFF64, nonTls
 nop
 .reloc ., R_X86_64_GOTTPOFF, nonTls
 nop
 .reloc ., R_X86_64_TLSGD, nonTls
 nop
## On error, handleTlsGd skips the next relocation (normally the
## __tls_get_addr call). Add a benign one so the following relocations
## are still scanned.
 .reloc ., R_X86_64_NONE
 nop
 .reloc ., R_X86_64_GOTPC32_TLSDESC, nonTls
 nop
 .reloc ., R_X86_64_DTPOFF64, nonTls
 nop
 .reloc ., R_X86_64_TLSLD, nonTls
 nop
## Likewise for handleTlsLd; this also keeps the ++it past-end walk
## in bounds.
 .reloc ., R_X86_64_NONE
 nop

.data
.type nonTls,@object
.globl nonTls
nonTls:
 .word 0
.size nonTls, 4
