# REQUIRES: x86
## Test relocations referencing symbols defined relative to sections discarded by /DISCARD/.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -T a.lds a.o b.o -z undefs -o /dev/null 2>&1 | count 0
# RUN: ld.lld -T a.lds a.o b.o -o /dev/null 2>&1 | count 0
# RUN: ld.lld -r -T a.lds a.o b.o -o /dev/null 2>&1 | count 0

#--- a.s
.globl _start
_start:

.section .aaa,"a"
.globl global, weakref1
.weak weak, weakref2
global:
weak:
weakref1:
weakref2:
  .quad 0

.section .bbb,"aw"
  .quad .aaa

#--- b.s
.weak weakref1, weakref2
.section .data,"aw"
  .quad global
  .quad weak
  .quad weakref1
  .quad weakref2

#--- a.lds
SECTIONS { /DISCARD/ : { *(.aaa) } }
