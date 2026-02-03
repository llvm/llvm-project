// this test checks a load literal instructions changed to movk

# REQUIRES: system-linux

# RUN: %clang %cflags %s -Wa,--defsym,CIBIGFUNC=1 -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:    --keep-nops --eliminate-unreachable=false \
# RUN:    | FileCheck %s --check-prefix=CHECK-LOGS
# RUN: llvm-objdump -d --disassemble-symbols=foo %t.bolt \
# RUN:    | FileCheck %s --check-prefix=CHECK-FOO

# RUN: %clang %cflags %s -Wa,--defsym,CIOUTSIDEFUNC=1 -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:    --keep-nops --eliminate-unreachable=false \
# RUN:    | FileCheck %s --check-prefix=CHECK-LOGS
# RUN: llvm-objdump -d --disassemble-symbols=foo %t.bolt \
# RUN:    | FileCheck %s --check-prefix=CHECK-FOO

# CHECK-LOGS: simplified 2 out of 2 loads

# CHECK-FOO: movk w23, #0x0, lsl #16
# CHECK-FOO-NEXT: movk w23, #0x64
# CHECK-FOO-NEXT: movk x24, #0x0, lsl #48
# CHECK-FOO-NEXT: movk x24, #0x0, lsl #32
# CHECK-FOO-NEXT: movk x24, #0x0, lsl #16
# CHECK-FOO-NEXT: movk x24, #0x3

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
// For AArch64, there is the issue related to emitting a constant
// island to the end of a function, ldr literal instruction can be
// out of available address range when the function size  is ~1MB.
// This strings are comment out due to unit test time execution.
// ~1 minute.
// .rep 0x100000
//    nop
// .endr
    b .Lreturn_point
.endif

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
# Dummy relocation to force relocation mode
.reloc 0, R_AARCH64_NONE
  mov x0, #0
  bl foo
  mov x0, 0
  mov w8, #93
  svc #0

.size main, .-main

