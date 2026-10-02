# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## TLS relocations in non-alloc sections are checked as well. llvm-mc marks
## symbols referenced through '@dtpoff' as STT_TLS, so use .reloc to create a
## reference that is genuinely non-STT_TLS.

# CHECK: error: relocation R_X86_64_DTPOFF64 against nonTls cannot be used with a non-STT_TLS symbol

.text
.globl _start
_start:
 nop

.section .debug_info,"",@progbits
 .reloc ., R_X86_64_DTPOFF64, nonTls
 .quad 0

.data
.globl nonTls
.type nonTls,@object
nonTls:
 .word 0
.size nonTls, 4
