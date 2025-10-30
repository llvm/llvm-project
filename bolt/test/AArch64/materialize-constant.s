// this test checks a load literal instructions changed to movk

// REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o

# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags -pie %t.o -o %t.exe -Wl,-q -Wl,-z,relro -Wl,-z,now
# RUN: llvm-bolt %t.exe -o %t.bolt -data %t.fdata \
# RUN:    --keep-nops --eliminate-unreachable=false
# RUN: llvm-objdump --disassemble-symbols=foo %t.bolt | FileCheck %s

# CHECK: mov{{.*}} w19, #0
# CHECK-NEXT: mov{{.*}} w22, #0
# CHECK-NEXT: movk{{.*}} w23, #0, lsl #16
# CHECK-NEXT: movk{{.*}} w23, #100
# CHECK-NEXT: movk{{.*}} w24, #0, lsl #16
# CHECK-NEXT: movk{{.*}} w24, #3

  .text
  .align 4
  .local foo
  .type foo, %function
foo:
# FDATA: 1 main 0 1 foo 0 0 10
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp

    mov w19, #0 // counter = 0
    mov w22, #0 // result = 0

    ldr w23, .Llimit
    ldr w24, .LStep
    b .LStub

.LConstants:
  .Llimit: .word 100
  .LStep:  .word 3

.LStub:
.rep 0x100000
    nop
.endr
    b .Lmain_loop

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


  .global main
  .type main, %function
main:
  mov x0, #0
  bl foo
  mov     x0, 0
  mov     w8, #93
  svc     #0

  .size main, .-main
