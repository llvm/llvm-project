# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .text %t.out | FileCheck %s
# CHECK: Hex dump of section '.text':
# CHECK-NEXT: 01020304

## --shuffle-sections= shuffles input sections.
# RUN: ld.lld --shuffle-sections=1 %t.o -o %t1.out
# RUN: llvm-readelf -x .text %t1.out | FileCheck %s --check-prefix=SHUFFLE1
# SHUFFLE1: Hex dump of section '.text':
# SHUFFLE1-NEXT: 0204cccc 0103

## Test that --shuffle-sections= can be used with --symbol-ordering-file
# RUN: echo "foo" > %t_order.txt
# RUN: echo "_start " >> %t_order.txt

# RUN: ld.lld --symbol-ordering-file %t_order.txt --shuffle-sections=2 %t.o -o %t2.out
# RUN: llvm-readelf -x .text %t2.out | FileCheck %s --check-prefix=SHUFFLE2
# SHUFFLE2: Hex dump of section '.text':
# SHUFFLE2-NEXT: 02cccccc 010304

# RUN: ld.lld --symbol-ordering-file %t_order.txt --shuffle-sections=3 %t.o -o %t3.out
# RUN: llvm-readelf -x .text %t3.out | FileCheck %s --check-prefix=SHUFFLE3
# SHUFFLE3: Hex dump of section '.text':
# SHUFFLE3-NEXT: 02cccccc 010403

## As a special case, -1 reverses sections as a stable transform.
# RUN: ld.lld --shuffle-sections=-1 %t.o -o %t-1.out
# RUN: llvm-readelf -x .text %t-1.out | FileCheck %s --check-prefix=SHUFFLE-1
# SHUFFLE-1: Hex dump of section '.text':
# SHUFFLE-1-NEXT: 040302cc 01

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
