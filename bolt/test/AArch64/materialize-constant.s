// this test checks a load literal instructions changed to movk

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym CIBIGFUNC=1 %s -o %t.o
# RUN: %clang %cflags %t.o -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:    --keep-nops --eliminate-unreachable=false \
# RUN:    | FileCheck %s --check-prefix=CHECK-LOGS

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --defsym CIOUTSIDEFUNC=1 %s -o %t.o
# RUN: %clang %cflags %t.o -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:    --keep-nops --eliminate-unreachable=false \
# RUN:    | FileCheck %s --check-prefix=CHECK-LOGS

# CHECK-LOGS: simplified 2 out of 2 loads

  .text
  .align 4
  .local foo
  .type foo, %function
foo:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp

    mov w19, #0 // counter = 0
    mov w22, #0 // result = 0

    ldr w23, .Llimit
    ldr x24, .LStep

.ifdef CIBIGFUNC
    b .LStub
.LConstants:
  .Llimit: .word 100
  .LStep:  .xword 3
.LStub:
.rep 0x100000
    nop
.endr
    b .Lmain_loop
.endif

.Lmain_loop:
    madd w22, w19, w24, w22  // result += counter * increment
    add w19, w19, #1
    cmp w19, w23
    b.lt .Lmain_loop
    mov w0, w22
    b .Lreturn_point
.Lreturn_point:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
.size foo, .-foo

.ifdef CIOUTSIDEFUNC
.LConstants:
  .Llimit: .word 100
  .LStep:  .xword 3
.endif


  .global main
  .type main, %function
main:
  mov x0, #0
  bl foo
  mov x0, 0
  mov w8, #93
  svc #0

.size main, .-main

