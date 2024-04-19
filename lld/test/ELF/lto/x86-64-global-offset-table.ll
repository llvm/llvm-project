; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

; RUN: cat a.ll medium.ll | llvm-as - -o medium.bc
; RUN: ld.lld -pie --no-relax medium.bc b.o -o medium
; RUN: llvm-objdump -d medium | FileCheck %s

; RUN: cat a.ll large.ll | llvm-as - -o large.bc
; RUN: ld.lld -pie large.bc b.o -o large
; RUN: llvm-objdump -d large | FileCheck %s

; RUN: cat a.ll medium.ll ref.ll | llvm-as - -o ref.bc
; RUN: ld.lld -pie --no-relax -u ref ref.bc b.o -o ref
; RUN: llvm-objdump -d ref | FileCheck %s

; CHECK: movabsq

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@i = external global i32

define dso_local void @_start() {
entry:
  %0 = load i32, ptr @i
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @i
  ret void
}

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 1, !"Large Data Threshold", i64 0}

;--- medium.ll
!3 = !{i32 1, !"Code Model", i32 3}

;--- large.ll
!3 = !{i32 1, !"Code Model", i32 4}

;--- ref.ll
@_GLOBAL_OFFSET_TABLE_ = external global [0 x i8]

define dso_local ptr @ref() {
entry:
  ret ptr @_GLOBAL_OFFSET_TABLE_
}

;--- b.s
.data
.globl i
i:
.long 0
