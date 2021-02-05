# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: llvm-objdump --syms --full-contents %t | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK: {{0*}}[[#%x, SUB1ADDR:]] g {{.*}} __DATA,subby _sub1
# CHECK: {{0*}}[[#%x, SUB2ADDR:]] g {{.*}} __DATA,subby _sub2
# CHECK-LABEL: Contents of section __DATA,subby:
# CHECK: [[#SUB1ADDR]] 10000000
# CHECK: [[#SUB2ADDR]] f0ffffff

.globl _main, _sub1, _sub2

.section __DATA,subby
L_.subtrahend_1:
  .space 16
L_.minuend_1:
  .space 16
L_.minuend_2:
  .space 16
L_.subtrahend_2:
  .space 16
_sub1:
  .long L_.minuend_1 - L_.subtrahend_1
  .space 12
_sub2:
  .long L_.minuend_2 - L_.subtrahend_2

.text
_main:
  mov $0, %rax
  ret
