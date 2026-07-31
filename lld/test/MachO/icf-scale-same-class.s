# REQUIRES: x86
# RUN: rm -rf %t*

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem --icf=all -o %t %t.o
# RUN: llvm-objdump --macho --section-headers %t | FileCheck %s --check-prefix=SECT
# RUN: llvm-objdump --macho --syms %t | FileCheck %s --check-prefix=SYMS

## Wall clock on single class, --icf=all:
##      N     re-walk   incremental
##   64 Ki      3.5 s        0.09 s
##  128 Ki     14.5 s        0.16 s
##  256 Ki     66.8 s        0.32 s
##  512 Ki    408.6 s        0.68 s

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

## Unlike icf-scale.s, every generated body here is identical.
.macro gen_4 c
  .globl _f0\c, _f1\c, _f2\c, _f3\c
  _f0\c:; movl $7, %eax; ret
  _f1\c:; movl $7, %eax; ret
  _f2\c:; movl $7, %eax; ret
  _f3\c:; movl $7, %eax; ret
.endm

.macro gen_16 c
  gen_4 0\c
  gen_4 1\c
  gen_4 2\c
  gen_4 3\c
.endm

.macro gen_64 c
  gen_16 0\c
  gen_16 1\c
  gen_16 2\c
  gen_16 3\c
.endm

.macro gen_256 c
  gen_64 0\c
  gen_64 1\c
  gen_64 2\c
  gen_64 3\c
.endm

.macro gen_1024 c
  gen_256 0\c
  gen_256 1\c
  gen_256 2\c
  gen_256 3\c
.endm

.macro gen_4096 c
  gen_1024 0\c
  gen_1024 1\c
  gen_1024 2\c
  gen_1024 3\c
.endm

.macro gen_16384 c
  gen_4096 0\c
  gen_4096 1\c
  gen_4096 2\c
  gen_4096 3\c
.endm

.macro gen_65536 c
  gen_16384 0\c
  gen_16384 1\c
  gen_16384 2\c
  gen_16384 3\c
.endm

.macro gen_262144 c
  gen_65536 0\c
  gen_65536 1\c
  gen_65536 2\c
  gen_65536 3\c
.endm

.globl _f_first
_f_first:; movl $7, %eax; ret

gen_262144 a
gen_262144 b

.globl _f_last
_f_last:; movl $7, %eax; ret

.globl _main
_main:
  ret
