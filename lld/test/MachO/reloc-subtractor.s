# REQUIRES: x86, aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/test.s -o %t/arm64.o
# RUN: %lld -lSystem %t/x86_64.o -o %t/x86_64 -order_file %t/order-file
# RUN: llvm-objdump --syms --full-contents --rebase %t/x86_64 | FileCheck %s
# RUN: %lld -arch arm64 -lSystem %t/arm64.o -o %t/arm64 -order_file %t/order-file
# RUN: llvm-objdump --syms --full-contents --rebase %t/arm64 | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       {{0*}}[[#%x, SUB1ADDR:]] l {{.*}} __DATA,bar _sub1
# CHECK:       {{0*}}[[#%x, SUB2ADDR:]] l {{.*}} __DATA,bar _sub2
# CHECK:       {{0*}}[[#%x, SUB3ADDR:]] l {{.*}} __DATA,bar _sub3
# CHECK:       {{0*}}[[#%x, SUB4ADDR:]] l {{.*}} __DATA,bar _sub4
# CHECK:       {{0*}}[[#%x, SUB5ADDR:]] l {{.*}} __DATA,bar _sub5
# CHECK-LABEL: Contents of section __DATA,bar:
# CHECK:       [[#SUB1ADDR]] 10000000
# CHECK-NEXT:  [[#SUB2ADDR]] f2ffffff
# CHECK-NEXT:  [[#SUB3ADDR]] 14000000 00000000
# CHECK-NEXT:  [[#SUB4ADDR]] f6ffffff ffffffff
# CHECK-NEXT:  [[#SUB5ADDR]] f1ffffff ffffffff
# CHECK:       Rebase table:
# CHECK-NEXT:  segment  section            address     type
# CHECK-EMPTY:

#--- test.s

.globl _main, _subtrahend_1, _subtrahend_2, _minuend1, _minuend2

.section __DATA,foo
  .space 16
L_.minuend:
  .space 16

.section __DATA,bar
_sub1:
  .long _minuend_1 - _subtrahend_1
  .space 12
_sub2:
  .long _minuend_2 - _subtrahend_2 + 2
  .space 12
_sub3:
  .quad _minuend_1 - _subtrahend_1 + 4
  .space 8
_sub4:
  .quad _minuend_2 - _subtrahend_2 + 6
  .space 8
_sub5:
  .quad L_.minuend - _subtrahend_1 + 1
  .space 8

_minuend_1:
  .space 16
_minuend_2:
  .space 16
_subtrahend_1:
  .space 16
_subtrahend_2:
  .space 16

.text
.p2align 2
_main:
  ret

.subsections_via_symbols

#--- order-file
## Reorder the symbols to make sure that the addends are being associated with
## the minuend (and not the subtrahend) relocation.
_subtrahend_1
_minuend_1
_minuend_2
_subtrahend_2
