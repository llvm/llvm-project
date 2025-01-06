# REQUIRES: arm

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=armv7a-none-linux-gnueabi %t/a.s -o %t1.o
# RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=armv7a-none-linux-gnueabi %t/b.s -o %t2.o
# RUN: ld.lld -shared %t1.o %t2.o -o %t.so
# RUN: llvm-objdump -d %t.so | FileCheck %s

## Check that, when the input is a mixture of objects which can and cannot use
## the ARM ISA, we use the default ARM PLT sequences.

# CHECK:      <.plt>:
# CHECK-NEXT: e52de004      str     lr, [sp, #-0x4]!
# CHECK-NEXT: e28fe600      add     lr, pc, #0, #12
# CHECK-NEXT: e28eea20      add     lr, lr, #32, #20
# CHECK-NEXT: e5bef084      ldr     pc, [lr, #0x84]!
# CHECK-NEXT: d4 d4 d4 d4   .word   0xd4d4d4d4
# CHECK-NEXT: d4 d4 d4 d4   .word   0xd4d4d4d4
# CHECK-NEXT: d4 d4 d4 d4   .word   0xd4d4d4d4
# CHECK-NEXT: d4 d4 d4 d4   .word   0xd4d4d4d4
# CHECK-NEXT: e28fc600      add     r12, pc, #0, #12
# CHECK-NEXT: e28cca20      add     r12, r12, #32, #20
# CHECK-NEXT: e5bcf06c      ldr     pc, [r12, #0x6c]!
# CHECK-NEXT: d4 d4 d4 d4   .word   0xd4d4d4d4

#--- a.s
  .globl foo
  .type foo, %function
  .globl bar
  .type bar, %function

  .thumb
foo:
  bl bar
  bx lr

#--- b.s
  .eabi_attribute Tag_ARM_ISA_use, 0

  .arm
  .globl bar
  .type bar, %function
bar:
  bx lr
