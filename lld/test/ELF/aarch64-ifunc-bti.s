# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %p/Inputs/aarch64-addrifunc.s -o %t1.o

# RUN: ld.lld --shared --soname=t1.so %t1.o -o %t1.so
# RUN: ld.lld --pie %t1.so %t.o -o %t
# RUN: llvm-objdump -d -mattr=+bti -triple=aarch64-linux-gnu %t | FileCheck %s

# When the address of an ifunc is taken using a non-got reference which clang
# can do, LLD exports a canonical PLT entry that may have its address taken so
# we must use bti c.

# CHECK: Disassembly of section .plt:
# CHECK: 0000000000010340 .plt:
# CHECK-NEXT:    10340: 5f 24 03 d5                     bti     c
# CHECK-NEXT:    10344: f0 7b bf a9                     stp     x16, x30, [sp, #-16]!
# CHECK-NEXT:    10348: 10 01 00 90                     adrp    x16, #131072
# CHECK-NEXT:    1034c: 11 5e 42 f9                     ldr     x17, [x16, #1208]
# CHECK-NEXT:    10350: 10 e2 12 91                     add     x16, x16, #1208
# CHECK-NEXT:    10354: 20 02 1f d6                     br      x17
# CHECK-NEXT:    10358: 1f 20 03 d5                     nop
# CHECK-NEXT:    1035c: 1f 20 03 d5                     nop
# CHECK: 0000000000010360 func1@plt:
# CHECK-NEXT:    10360: 5f 24 03 d5                     bti     c
# CHECK-NEXT:    10364: 10 01 00 90                     adrp    x16, #131072
# CHECK-NEXT:    10368: 11 62 42 f9                     ldr     x17, [x16, #1216]
# CHECK-NEXT:    1036c: 10 02 13 91                     add     x16, x16, #1216
# CHECK-NEXT:    10370: 20 02 1f d6                     br      x17
# CHECK-NEXT:    10374: 1f 20 03 d5                     nop
# CHECK-NEXT:           ...
# CHECK: 0000000000010380 myfunc:
# CHECK-NEXT:    10380: 5f 24 03 d5                     bti     c
# CHECK-NEXT:    10384: 10 01 00 90                     adrp    x16, #131072
# CHECK-NEXT:    10388: 11 66 42 f9                     ldr     x17, [x16, #1224]
# CHECK-NEXT:    1038c: 10 22 13 91                     add     x16, x16, #1224
# CHECK-NEXT:    10390: 20 02 1f d6                     br      x17
# CHECK-NEXT:    10394: 1f 20 03 d5                     nop

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

.text
.globl myfunc
.type myfunc,@gnu_indirect_function
myfunc:
 ret

.globl func1

.text
.globl _start
.type _start, %function
_start:
  bl func1
  adrp x8, myfunc
  add x8, x8, :lo12:myfunc
  ret
