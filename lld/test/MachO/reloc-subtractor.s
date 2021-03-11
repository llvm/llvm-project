# REQUIRES: x86, aarch64
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/arm64.o
# RUN: %lld -lSystem %t/x86_64.o -o %t/x86_64
# RUN: llvm-objdump --syms --full-contents --rebase %t/x86_64 | FileCheck %s
# RUN: %lld -arch arm64 -lSystem %t/arm64.o -o %t/arm64
# RUN: llvm-objdump --syms --full-contents --rebase %t/arm64 | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       {{0*}}[[#%x, SUB1ADDR:]] l {{.*}} __DATA,__data _sub1
# CHECK:       {{0*}}[[#%x, SUB2ADDR:]] l {{.*}} __DATA,__data _sub2
# CHECK:       {{0*}}[[#%x, SUB3ADDR:]] l {{.*}} __DATA,__data _sub3
# CHECK:       {{0*}}[[#%x, SUB4ADDR:]] l {{.*}} __DATA,__data _sub4
# CHECK-LABEL: Contents of section __DATA,__data:
# CHECK:       [[#SUB1ADDR]] 10000000
# CHECK-NEXT:  [[#SUB2ADDR]] f2ffffff
# CHECK-NEXT:  [[#SUB3ADDR]] 14000000 00000000
# CHECK-NEXT:  [[#SUB4ADDR]] f6ffffff ffffffff
# CHECK:       Rebase table:
# CHECK-NEXT:  segment  section            address     type
# CHECK-EMPTY:

.globl _main, _subtrahend_1, _subtrahend_2, _minuend1, _minuend2

.data
_subtrahend_1:
  .space 16
_minuend_1:
  .space 16
_minuend_2:
  .space 16
_subtrahend_2:
  .space 16
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

.text
.p2align 2
_main:
  ret
