; REQUIRES: x86
; RUN: llvm-as %s -o %t.bc
; RUN: ld.lld %t.bc -o %t

target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

define dso_local void @f() {
entry:
  ret void
}

define dso_local void @_start() {
entry:
  call void @f()
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
