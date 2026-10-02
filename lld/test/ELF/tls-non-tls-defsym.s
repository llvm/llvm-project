# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --defsym defsymSym=42 %t.o -o /dev/null 2>&1 | count 0

## Script-defined symbols have no ELF type, so the STT_TLS invariant cannot
## be enforced on them.

.text
.globl _start
_start:
 .reloc ., R_X86_64_TPOFF64, defsymSym
 nop
