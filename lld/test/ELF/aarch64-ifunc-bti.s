# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %p/Inputs/aarch64-addrifunc.s -o %t1.o

# RUN: ld.lld --shared --soname=t1.so %t1.o -o %t1.so
# RUN: ld.lld --pie %t1.so %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+bti --triple=aarch64-linux-gnu %t | FileCheck %s

# When the address of an ifunc is taken using a non-got reference which clang
# can do, LLD exports a canonical PLT entry that may have its address taken so
# we must use bti c.

# CHECK: Disassembly of section .plt:
# CHECK: 0000000000010380 <.plt>:
# CHECK-NEXT:    10380:         bti     c
# CHECK-NEXT:                   stp     x16, x30, [sp, #-16]!
# CHECK-NEXT:                   adrp    x16, #131072
# CHECK-NEXT:                   ldr     x17, [x16, #1288]
# CHECK-NEXT:                   add     x16, x16, #1288
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop
# CHECK-NEXT:                   nop
# CHECK: 00000000000103a0 <func1@plt>:
# CHECK-NEXT:    103a0:         bti     c
# CHECK-NEXT:                   adrp    x16, #131072
# CHECK-NEXT:                   ldr     x17, [x16, #1296]
# CHECK-NEXT:                   add     x16, x16, #1296
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop
# CHECK-EMPTY:
# CHECK: Disassembly of section .iplt:
# CHECK-EMPTY:
# CHECK-NEXT: 00000000000103c0 <myfunc>:
# CHECK-NEXT:    103c0:         bti     c
# CHECK-NEXT:                   adrp    x16, #131072
# CHECK-NEXT:                   ldr     x17, [x16, #1304]
# CHECK-NEXT:                   add     x16, x16, #1304
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop

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
