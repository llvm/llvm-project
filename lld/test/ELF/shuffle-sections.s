# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .text %t.out | FileCheck %s
# CHECK: Hex dump of section '.text':
# CHECK-NEXT: 01020304

## --shuffle-sections= shuffles input sections.
# RUN: ld.lld --shuffle-sections='*=1' %t.o -o %t1.out
# RUN: llvm-readelf -x .text %t1.out | FileCheck %s --check-prefix=SHUFFLE1
# SHUFFLE1: Hex dump of section '.text':
# SHUFFLE1-NEXT: 030402cc 01

## Test that --shuffle-sections= can be used with --symbol-ordering-file
# RUN: echo "foo" > %t_order.txt
# RUN: echo "_start " >> %t_order.txt

# RUN: ld.lld --symbol-ordering-file %t_order.txt --shuffle-sections='*=2' %t.o -o %t2.out
# RUN: llvm-readelf -x .text %t2.out | FileCheck %s --check-prefix=SHUFFLE2
# SHUFFLE2: Hex dump of section '.text':
# SHUFFLE2-NEXT: 02cccccc 010304

# RUN: ld.lld -z norelro --symbol-ordering-file %t_order.txt --shuffle-sections='*=3' %t.o -o %t3.out
# RUN: llvm-readelf -x .text %t3.out | FileCheck %s --check-prefix=SHUFFLE3
# SHUFFLE3: Hex dump of section '.text':
# SHUFFLE3-NEXT: 02cccccc 010403

## As a special case, -1 reverses sections as a stable transform.
# RUN: ld.lld --shuffle-sections '*=-1' %t.o -o %t-1.out
# RUN: llvm-readelf -x .text %t-1.out | FileCheck %s --check-prefix=SHUFFLE-1
# SHUFFLE-1: Hex dump of section '.text':
# SHUFFLE-1-NEXT: 040302cc 01

## .text does not change its order while .text.{foo,bar,zed} are reversed.
# RUN: ld.lld --shuffle-sections '.text.*=-1' %t.o -o %t4.out
# RUN: llvm-readelf -x .text %t4.out | FileCheck %s --check-prefix=SHUFFLE4
# SHUFFLE4: Hex dump of section '.text':
# SHUFFLE4-NEXT: 01040302

## Reversing twice restores the original order.
# RUN: ld.lld --shuffle-sections '.text.*=-1' --shuffle-sections '.text.*=-1' %t.o -o %t.out
# RUN: llvm-readelf -x .text %t.out | FileCheck %s

## Test all possible invalid cases.
# RUN: not ld.lld --shuffle-sections= 2>&1 | FileCheck %s --check-prefix=USAGE -DV=
# RUN: not ld.lld --shuffle-sections=a= 2>&1 | FileCheck %s --check-prefix=USAGE -DV=a=
# RUN: not ld.lld --shuffle-sections==0 2>&1 | FileCheck %s --check-prefix=USAGE -DV==0
# RUN: not ld.lld --shuffle-sections=a 2>&1 | FileCheck %s --check-prefix=USAGE -DV=a

# USAGE: error: --shuffle-sections=: expected <section_glob>=<seed>, but got '[[V]]'

# RUN: not ld.lld --shuffle-sections='['=0 2>&1 | FileCheck %s --check-prefix=INVALID

# INVALID: error: --shuffle-sections=: invalid glob pattern, unmatched '[': [

## .text has an alignment of 4.
.global _start
_start:
  .byte 1

.section .text.foo,"ax"
.global foo
foo:
  .byte 2

.section .text.bar,"ax"
.global bar
bar:
  .byte 3

.section .text.zed,"ax"
.global zed
zed:
  .byte 4
