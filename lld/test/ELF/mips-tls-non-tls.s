# REQUIRES: mips
# RUN: llvm-mc -filetype=obj -triple=mips %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=TEXT --implicit-check-not=error:
# RUN: llvm-mc -filetype=obj -triple=mips %s -defsym=DEBUGONLY=1 -o %t.debug.o
# RUN: not ld.lld %t.debug.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG --implicit-check-not=error:

## TLS relocations can only reference symbols with type STT_TLS (gABI).
## llvm-mc marks symbols referenced through TLS syntax as STT_TLS, so use
## .reloc to create references that are genuinely non-STT_TLS.

# TEXT: error: relocation R_MIPS_TLS_GD against nonTls cannot be used with a non-STT_TLS symbol
# TEXT: error: relocation R_MIPS_TLS_LDM against nonTls cannot be used with a non-STT_TLS symbol
# TEXT: error: relocation R_MIPS_TLS_TPREL_HI16 against nonTls cannot be used with a non-STT_TLS symbol

## R_MIPS_TLS_GOTTPREL is classified RE_MIPS_GOT_OFF, which is shared with
## non-TLS GOT relocations, so it is not covered by the central check.

# DEBUG: error: relocation R_MIPS_TLS_DTPREL32 against nonTls cannot be used with a non-STT_TLS symbol

.ifndef DEBUGONLY
.text
.globl _start
_start:
 .reloc ., R_MIPS_TLS_GD, nonTls
 nop
 .reloc ., R_MIPS_TLS_LDM, nonTls
 nop
 .reloc ., R_MIPS_TLS_TPREL_HI16, nonTls
 nop
.else
## .dtprelword does not retype the symbol, and mc does not recognize the
## R_MIPS_TLS_DTPREL32 name in .reloc, so use it to test the non-alloc path.
.section .debug_info,"",@progbits
 .dtprelword nonTls
.endif

.data
.globl nonTls
nonTls:
 .word 0
