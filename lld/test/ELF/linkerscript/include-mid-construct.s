# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

## A stray ';' in the parent after INCLUDE cannot complete the inner assignment.
# RUN: not ld.lld a.o -T top.lds 2>&1 | FileCheck %s --check-prefix=TOP
# TOP: error: inc-top.lds:1: unexpected EOF

#--- top.lds
INCLUDE "inc-top.lds";

#--- inc-top.lds
foo = 1

#--- a.s
.globl _start
_start:
  ret
