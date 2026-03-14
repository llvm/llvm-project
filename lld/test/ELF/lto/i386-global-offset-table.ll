; REQUIRES: x86
;; LTO-generated relocatable files may reference _GLOBAL_OFFSET_TABLE_ while
;; the IR does not mention _GLOBAL_OFFSET_TABLE_.
;; Test that there is no spurious "undefined symbol" error.

; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: llvm-as %s -o a.bc
; RUN: ld.lld -pie a.bc -o a
; RUN: llvm-nm a | FileCheck %s

; CHECK: d _GLOBAL_OFFSET_TABLE_

target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

@i = global i32 0

define dso_local void @_start() {
entry:
  %0 = load i32, ptr @i
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @i
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
