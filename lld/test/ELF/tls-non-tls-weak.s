# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null 2>&1 | count 0

## A weak undefined symbol's type is not known until resolution, so the
## STT_TLS invariant cannot be enforced on it. GNU ld accepts this as well.

.text
.globl _start
_start:
 .reloc ., R_X86_64_TPOFF64, undefSym
 nop

.weak undefSym
