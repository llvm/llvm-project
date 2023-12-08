; REQUIRES: x86
; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; RUN: rm -f %t1.o.4.opt.bc
; RUN: ld.lld --lto-sample-profile=%p/Inputs/sample-profile.prof %t1.o %t2.o --save-temps -o %t3
; RUN: opt -S %t1.o.4.opt.bc | FileCheck %s

; RUN: rm -f %t1.o.4.opt.bc
; RUN: ld.lld --plugin-opt=sample-profile=%p/Inputs/sample-profile.prof %t1.o %t2.o --save-temps -o %t3
; RUN: opt -S %t1.o.4.opt.bc | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: ![[#]] = !{i32 1, !"ProfileSummary", ![[#]]}
declare void @g(...)
declare void @h(...)

define void @f() {
entry:
  call void (...) @g()
  call void (...) @h()
  ret void
}
