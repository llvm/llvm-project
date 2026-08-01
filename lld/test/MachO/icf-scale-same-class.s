# REQUIRES: x86
# RUN: rm -rf %t*

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem --icf=all -o %t %t.o
# RUN: llvm-objdump --macho --section-headers %t | FileCheck %s --check-prefix=SECT
# RUN: llvm-objdump --macho --syms %t | FileCheck %s --check-prefix=SYMS

## Every body folds into one, so __text holds a single 6-byte body plus _main
## rather than the 3 MiB (0x30000d) the unfolded copies occupy.
# SECT:      Idx Name          Size
# SECT-NEXT: 0 __text        00000009

## The sentinels bracket the group, so checking that the first and last members
## of the class share an address covers the whole run of folds.
# SYMS-DAG: [[#%.16x,F:]] g     F __TEXT,__text _f_first
# SYMS-DAG: [[#F]]        g     F __TEXT,__text _f_last

.subsections_via_symbols
.text
.p2align 2

.globl _f_first
_f_first:; movl $7, %eax; ret

## Unlike icf-scale.s, every body generated here is identical, so all 512 Ki
## land in one equivalence class. \+ iterates from 0 to n-1. $$7 is an escape
## for a literal $7: .rept expands its body as a macro with no parameters, and
## in one of those the Darwin assembler reads $7 as positional argument 7 and
## drops it.
.rept 524288
  .globl _f\+
  _f\+:; movl $$7, %eax; ret
.endr

.globl _f_last
_f_last:; movl $7, %eax; ret

.globl _main
_main:
  ret
