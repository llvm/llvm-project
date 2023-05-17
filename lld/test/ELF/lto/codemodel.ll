; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %ts -mllvm -code-model=small
; RUN: ld.lld %t.o -o %tl -mllvm -code-model=large
; RUN: llvm-objdump --no-print-imm-hex -d %ts | FileCheck %s --check-prefix=CHECK-SMALL
; RUN: llvm-objdump --no-print-imm-hex -d %tl | FileCheck %s --check-prefix=CHECK-LARGE

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@data = internal constant [0 x i32] []

define ptr @_start() nounwind readonly {
entry:
; CHECK-SMALL-LABEL:  <_start>:
; CHECK-SMALL: movl    $2097440, %eax
; CHECK-LARGE-LABEL: <_start>:
; CHECK-LARGE: movabsq $2097440, %rax
    ret ptr @data
}
