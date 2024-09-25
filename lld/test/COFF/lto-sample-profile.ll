; REQUIRES: x86
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; RUN: rm -f %t1.o.4.opt.bc
; RUN: lld-link /lto-sample-profile:%p/Inputs/lto-sample-profile.prof /lldsavetemps /entry:f /subsystem:console %t1.o %t2.o /out:%t3.exe
; RUN: opt -S %t1.o.4.opt.bc | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; CHECK: ![[#]] = !{i32 1, !"ProfileSummary", ![[#]]}
declare void @g(...)

define void @h() {
  ret void
}
define void @f() {
entry:
  call void (...) @g()
  call void (...) @h()
  ret void
}
