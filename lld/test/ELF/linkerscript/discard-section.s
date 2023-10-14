# REQUIRES: x86
## Test relocations referencing symbols defined relative to sections discarded by /DISCARD/.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo "SECTIONS { /DISCARD/ : { *(.aaa*) } }" > %t.lds
# RUN: ld.lld -T %t.lds %t.o -z undefs -o /dev/null 2>&1 | count 0
# RUN: ld.lld -T %t.lds %t.o -o /dev/null 2>&1 | count 0
# RUN: ld.lld -r -T %t.lds %t.o -o /dev/null 2>&1 | count 0

.globl _start
_start:

.section .aaa,"a"
.globl global
.weak weak
global:
weak:
  .quad 0

.section .zzz,"a"
  .quad .aaa
  .quad global
  .quad weak
