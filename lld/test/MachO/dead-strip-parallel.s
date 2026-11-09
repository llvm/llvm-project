# REQUIRES: x86
## The symbol table here spans several chunks of the parallel root scan. Check
## that one and eight threads mark the same sections live.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: %lld -dead_strip --threads=1 %t.o -o %t1 -final_output out
# RUN: %lld -dead_strip --threads=8 %t.o -o %t8 -final_output out
# RUN: cmp %t1 %t8
# RUN: llvm-nm %t8 | grep _live | count 1000
# RUN: llvm-nm %t8 | grep _callee | count 1000
# RUN: llvm-nm %t8 | FileCheck %s --implicit-check-not=_dead
# CHECK: _main

.text
.globl _main
_main:
  ret

.macro funcs
.globl _live\@, _dead\@
.no_dead_strip _live\@
_live\@:
  callq _callee\@
  ret
_callee\@:
  ret
_dead\@:
  ret
.endm

.rept 1000
funcs
.endr

.subsections_via_symbols
