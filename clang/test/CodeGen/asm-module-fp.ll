; REQUIRES: arm-registered-target

; Check that module-level asm with FP instructions works when the target
; features are not implied by the architecture.

; RUN: %clang_cc1 -triple armv7r-unknown-none-eabi -target-feature +vfp4 -emit-llvm-bc %s -o %t.o

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7r-unknown-none-eabi"

module asm ".globl fp_add"
module asm "fp_add:"
module asm "vadd.f32 s0, s0, s1"
module asm "bx lr"
