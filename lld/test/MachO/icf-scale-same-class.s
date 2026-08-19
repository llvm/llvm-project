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

## 64 Ki identical function landing in one equivalence class.
.rept 65536
  .globl _f\+
  _f\+:; movl $$7, %eax; ret
.endr

.globl _f_last
_f_last:; movl $7, %eax; ret

.globl _main
_main:
  ret
